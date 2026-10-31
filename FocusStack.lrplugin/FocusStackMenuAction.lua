local LrTasks = import 'LrTasks'
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrExportSession = import 'LrExportSession'

local function debugLog(msg)
    local f = io.open("/tmp/focus_stack_import.log", "a")
    if f then
        f:write(os.date("%Y-%m-%d %H:%M:%S") .. " [MenuAction] " .. tostring(msg) .. "\n")
        f:close()
    end
end

LrTasks.startAsyncTask(function()
    local catalog = LrApplication.activeCatalog()
    local targetPhotos = catalog:getTargetPhotos()

    if not targetPhotos or #targetPhotos < 2 then
        LrDialogs.message(
            "Open Focus Stack",
            "Please select at least 2 bracketed photos in the Lightroom grid to focus stack.",
            "info"
        )
        return
    end

    debugLog("==========================================")
    debugLog("FocusStackMenuAction launched with " .. tostring(#targetPhotos) .. " target photos")

    -- Extract paths outside table.sort to prevent yielding across C-call boundary
    local photoPaths = {}
    for _, photo in ipairs(targetPhotos) do
        local p = ""
        pcall(function() photo.catalog:withReadAccessDo(function() p = photo:getRawMetadata('path') end) end)
        photoPaths[photo] = p
    end
    table.sort(targetPhotos, function(a, b)
        return (photoPaths[a] or "") < (photoPaths[b] or "")
    end)

    local firstPhoto = targetPhotos[1]
    local firstRawPath = photoPaths[firstPhoto]

    local expectedDir = firstRawPath and LrPathUtils.parent(firstRawPath)
    local expectedBase = firstRawPath and LrPathUtils.removeExtension(LrPathUtils.leafName(firstRawPath))
    local firstExt = firstRawPath and string.lower(LrPathUtils.extension(firstRawPath) or "")
    local isJpeg = (firstExt == "jpg" or firstExt == "jpeg")
    local outExt = isJpeg and ("." .. firstExt) or ".tif"
    local expectedPath = expectedDir and expectedBase and LrPathUtils.child(expectedDir, expectedBase .. "_stacked" .. outExt)
    if expectedPath and not LrFileUtils.exists(expectedPath) then
        local altExt = isJpeg and ".tif" or ".jpg"
        local altPath = expectedDir and expectedBase and LrPathUtils.child(expectedDir, expectedBase .. "_stacked" .. altExt)
        if altPath and LrFileUtils.exists(altPath) then
            expectedPath = altPath
        end
    end

    debugLog("expectedPath from targetPhotos: " .. tostring(expectedPath))

    local needRender = true
    if expectedPath and LrFileUtils.exists(expectedPath) then
        local btn = LrDialogs.confirm(
            "Stacked Photo Already Exists",
            "A 16-bit stacked photo is already present on disk in your photoshoot folder:\n" .. LrPathUtils.leafName(expectedPath) .. "\n\nWould you like to register it directly into your catalog now, or re-render a fresh stack?",
            "Register into Catalog",
            "Cancel",
            "Re-render Fresh Stack"
        )
        if btn == "cancel" then
            debugLog("User cancelled via existing photo prompt.")
            return
        elseif btn == "ok" then
            needRender = false
            debugLog("User selected Register into Catalog for existing file: " .. tostring(expectedPath))
        else
            needRender = true
            debugLog("User selected Re-render Fresh Stack.")
        end
    end

    local exportSettings = {
        LR_exportServiceProvider = "com.antigravity.openfocusstack",
        LR_exportServiceProviderTitle = "Open Focus Stack",
        LR_format = "TIFF",
        LR_export_format = "TIFF",
        LR_export_colorSpace = "ProPhotoRGB",
        LR_export_bitDepth = "16",
        LR_export_tiff_compression = "NONE",
        LR_export_destinationType = "sourceFolder",
        LR_export_useSubfolder = true,
        LR_export_subfolderName = ".focus_stack_tmp",
        LR_collisionHandling = "rename",
        reimportExportedPhoto = false,
        LR_reimportExportedPhoto = false,
        LR_addToCatalog = false,
        reimport_stackWithOriginal = false,
        outputSuffix = "_stacked",
        usePythonEngine = true,
        interactiveMode = true,
        batchSize = "8",
        deleteIntermediates = true,
        stackAboveOriginal = true,
        fullResolutionAlign = true,
        globalAlign = true,
        preserveExposure = true,
        invokedFromMenuAction = true,
    }

    if needRender then
        local exportSession = LrExportSession({
            photosToExport = targetPhotos,
            exportSettings = exportSettings,
        })

        debugLog("Starting exportSession:doExportOnCurrentTask()")
        exportSession:doExportOnCurrentTask()
        debugLog("exportSession:doExportOnCurrentTask() completed!")

        -- The export session has completed and released all internal locks.
        -- Settle filesystem writes briefly
        LrTasks.sleep(0.5)
    end

    -- Locate generated stacked file on disk
    local outputPath = expectedPath
    if not (outputPath and LrFileUtils.exists(outputPath)) then
        if expectedDir then
            for _, photo in ipairs(targetPhotos) do
                local p = photoPaths[photo]
                if p and #p > 0 then
                    local base = LrPathUtils.removeExtension(LrPathUtils.leafName(p))
                    local candidate = LrPathUtils.child(expectedDir, base .. "_stacked.tif")
                    if LrFileUtils.exists(candidate) then
                        outputPath = candidate
                        break
                    end
                end
            end
        end
    end

    debugLog("Post-export resolved outputPath: " .. tostring(outputPath))

    if not (outputPath and LrFileUtils.exists(outputPath)) then
        debugLog("Stacked output file not found on disk after export session.")
        return
    end

    -- Import the stacked photo into catalog now that the export session is 100% closed
    local finalPhoto = nil
    local addErr = nil

    catalog:withReadAccessDo(function()
        finalPhoto = catalog:findPhotoByPath(outputPath)
    end)

    local function tryAddPhoto()
        local res, err = nil, nil
        if firstPhoto then
            local ok, sRes, sMsg = pcall(function()
                return catalog:addPhoto(outputPath, firstPhoto, "above")
            end)
            if ok and sRes then
                debugLog("catalog:addPhoto succeeded with stack 'above' original!")
                return sRes, nil
            else
                debugLog("Stacking with original failed (" .. tostring(sMsg or sRes) .. "), trying standalone addPhoto...")
            end
        end
        return catalog:addPhoto(outputPath)
    end

    if not finalPhoto then
        debugLog("Attempting catalog ingestion outside export session, hasWriteAccess=" .. tostring(catalog.hasWriteAccess))
        if catalog.hasWriteAccess then
            local res, err = tryAddPhoto()
            debugLog("Direct addPhoto returned res=" .. tostring(res) .. ", err=" .. tostring(err))
            if not res and err then
                addErr = tostring(err)
            end
        else
            local writeStatus, writeMsg = catalog:withWriteAccessDo("Import Stacked Photo", function()
                local res, err = tryAddPhoto()
                debugLog("Inside withWriteAccessDo: addPhoto returned res=" .. tostring(res) .. ", err=" .. tostring(err))
                if not res and err then
                    addErr = tostring(err)
                end
            end, { timeout = 30 })

            debugLog("withWriteAccessDo returned writeStatus=" .. tostring(writeStatus) .. ", writeMsg=" .. tostring(writeMsg))

            if writeStatus ~= "executed" and writeStatus ~= "queued" then
                addErr = "Catalog write " .. tostring(writeStatus) .. ": " .. tostring(writeMsg or "lock busy")
            end
        end
    end

    -- Always retrieve actual LrPhoto instance via findPhotoByPath
    catalog:withReadAccessDo(function()
        finalPhoto = catalog:findPhotoByPath(outputPath)
    end)

    debugLog("Final imported photo object: " .. tostring(finalPhoto))

    if finalPhoto then
        -- Navigate grid view to containing folder and highlight stacked photo
        local folderDir = LrPathUtils.parent(outputPath)
        local folder = nil
        catalog:withReadAccessDo(function()
            folder = catalog:getFolderByPath(folderDir)
            if not folder then
                folder = catalog:getFolderByPath(folderDir:gsub("[/\\]+$", ""))
            end
        end)
        if folder then
            catalog:setActiveSources({ folder })
        else
            catalog:setActiveSources(catalog.kAllPhotos)
        end

        LrTasks.sleep(0.3)
        catalog:setSelectedPhotos(finalPhoto, {})
        -- Stacked and imported successfully. Silent on success per user preference.
    else
        LrDialogs.message(
            "Focus Stacking Import Diagnostics",
            "The 16-bit stacked photo was saved to disk:\n" .. outputPath .. "\n\n" ..
            "Lightroom catalog registration reported:\n" ..
            tostring(addErr or "photo not returned by catalog") .. "\n\n" ..
            "Full log written to: /tmp/focus_stack_import.log",
            "critical"
        )
    end
end)
