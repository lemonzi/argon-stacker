local LrTasks = import 'LrTasks'
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'

LrTasks.startAsyncTask(function()
    local catalog = LrApplication.activeCatalog()
    local candidatePaths = {}

    -- Check if user has photos selected in grid
    local targetPhotos = catalog:getTargetPhotos()
    if targetPhotos and #targetPhotos > 0 then
        for _, photo in ipairs(targetPhotos) do
            local rawPath = nil
            LrTasks.pcall(function()
                photo.catalog:withReadAccessDo(function()
                    rawPath = photo:getRawMetadata('path')
                end)
            end)
            if rawPath then
                local dir = LrPathUtils.parent(rawPath)
                local base = LrPathUtils.removeExtension(LrPathUtils.leafName(rawPath))
                table.insert(candidatePaths, LrPathUtils.child(dir, base .. "_stacked.tif"))
                table.insert(candidatePaths, LrPathUtils.child(dir, base .. "_stacked.tiff"))
            end
        end
    end

    local targetPath = nil
    for _, path in ipairs(candidatePaths) do
        local std = LrPathUtils.standardizePath(path)
        if LrFileUtils.exists(std) then
            targetPath = std
            break
        end
    end

    if not targetPath then
        local chosen = LrDialogs.runOpenPanel({
            title = "Select Stacked Photo to Ingest",
            canChooseFiles = true,
            canChooseDirectories = false,
            allowsMultipleSelection = false,
            fileTypes = { "tif", "tiff", "jpg", "jpeg" },
        })
        if chosen and #chosen > 0 then
            targetPath = LrPathUtils.standardizePath(chosen[1])
        end
    end

    if not targetPath or not LrFileUtils.exists(targetPath) then
        return
    end

    -- Check if photo is already indexed in catalog
    local existingPhoto = nil
    LrTasks.pcall(function()
        catalog:withReadAccessDo(function()
            existingPhoto = catalog:findPhotoByPath(targetPath)
        end)
    end)

    if existingPhoto then
        -- Photo is already registered; switch to its folder and highlight it
        local folderDir = LrPathUtils.parent(targetPath)
        LrTasks.pcall(function()
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
        end)
        LrTasks.sleep(0.3)
        LrTasks.pcall(function()
            catalog:setSelectedPhotos(existingPhoto, {})
        end)

        LrDialogs.message(
            "Photo Already in Catalog",
            "The photo is already in your catalog and has now been highlighted in the grid:\n" .. targetPath,
            "info"
        )
        return
    end

    -- Add photo to catalog inside withWriteAccessDo
    local addErr = nil
    local writeStatus = nil
    local writeMsg = nil
    local pcallOk, pcallErr = LrTasks.pcall(function()
        writeStatus, writeMsg = catalog:withWriteAccessDo("Register Stacked Photo", function()
            local res, err = catalog:addPhoto(targetPath)
            if not res and err then
                addErr = tostring(err)
            end
        end, { timeout = 30 })
    end)

    -- Always retrieve actual LrPhoto instance via findPhotoByPath
    local finalPhoto = nil
    LrTasks.pcall(function()
        catalog:withReadAccessDo(function()
            finalPhoto = catalog:findPhotoByPath(targetPath)
        end)
    end)

    if finalPhoto then
        local folderDir = LrPathUtils.parent(targetPath)
        LrTasks.pcall(function()
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
        end)
        LrTasks.sleep(0.3)
        LrTasks.pcall(function()
            catalog:setSelectedPhotos(finalPhoto, {})
        end)

        LrDialogs.message(
            "Registration Successful",
            "Successfully added stacked photo to your Lightroom catalog:\n" .. targetPath,
            "info"
        )
    else
        local errMsg = ""
        if not pcallOk then
            errMsg = "pcall error: " .. tostring(pcallErr)
        elseif writeStatus ~= "executed" and writeStatus ~= "queued" then
            errMsg = "Write access " .. tostring(writeStatus) .. ": " .. tostring(writeMsg or "locked by Lightroom")
        elseif addErr then
            errMsg = "addPhoto rejected: " .. addErr
        else
            errMsg = "addPhoto completed but photo not found in catalog indexing."
        end

        LrDialogs.message(
            "Registration Failed",
            "Could not register photo:\n" .. targetPath .. "\n\nDetails:\n" .. errMsg,
            "critical"
        )
    end
end)
