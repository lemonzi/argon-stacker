local LrTasks = import 'LrTasks'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrView = import 'LrView'
local LrColor = import 'LrColor'
local LrShell = import 'LrShell'

local exportServiceProvider = {}

-- Locate the focus-stack binary automatically relative to the plugin folder
local function getBinaryPath()
    if _PLUGIN and _PLUGIN.path then
        local bundled = LrPathUtils.child(_PLUGIN.path, "bin/focus-stack/focus-stack.app/Contents/MacOS/focus-stack")
        if LrFileUtils.exists(bundled) then
            return bundled
        end
    end
    if LrFileUtils.exists("/usr/local/bin/focus-stack") then
        return "/usr/local/bin/focus-stack"
    end
    if LrFileUtils.exists("/opt/homebrew/bin/focus-stack") then
        return "/opt/homebrew/bin/focus-stack"
    end
    return nil
end

-- Locate python3 on the user system
local function getPythonPath()
    if LrFileUtils.exists("/usr/bin/python3") then
        return "/usr/bin/python3"
    end
    if LrFileUtils.exists("/opt/homebrew/bin/python3") then
        return "/opt/homebrew/bin/python3"
    end
    if LrFileUtils.exists("/usr/local/bin/python3") then
        return "/usr/local/bin/python3"
    end
    return "python3"
end

-- Locate orientation tag helper script
local function getFixOrientationScriptPath()
    if _PLUGIN and _PLUGIN.path then
        local p = LrPathUtils.child(_PLUGIN.path, "bin/fix_orientation.py")
        if LrFileUtils.exists(p) then
            return p
        end
    end
    return nil
end

-- Locate 16-Bit Python Focus Stacking Engine (Self-Contained inside Plugin Bundle)
local function getPythonEnginePaths()
    local pluginDir = nil
    if _PLUGIN and _PLUGIN.path then
        pluginDir = _PLUGIN.path
    else
        local info = debug.getinfo(1, "S")
        if info and info.source and info.source:sub(1, 1) == "@" then
            pluginDir = LrPathUtils.parent(info.source:sub(2))
        end
    end

    if pluginDir then
        local venvPy = LrPathUtils.child(pluginDir, "bin/venv/bin/python3")
        local cliPy = LrPathUtils.child(pluginDir, "bin/py_focus_stack/cli.py")
        if LrFileUtils.exists(venvPy) and LrFileUtils.exists(cliPy) then
            return venvPy, cliPy
        end
    end
    return nil, nil
end

-- Automatically install the 1-Click Context Menu Preset into Lightroom User Presets
local function ensurePresetInstalled()
    local appData = LrPathUtils.getStandardFilePath('appData')
    if appData and _PLUGIN and _PLUGIN.path then
        local targetDir = LrPathUtils.child(appData, "Export Presets/User Presets")
        local targetFile = LrPathUtils.child(targetDir, "Open Focus Stack.lrtemplate")
        local sourceFile = LrPathUtils.child(_PLUGIN.path, "Export Presets/Open Focus Stack.lrtemplate")
        if LrFileUtils.exists(sourceFile) then
            if not LrFileUtils.exists(targetDir) then
                LrFileUtils.createAllDirectories(targetDir)
            end
            if LrFileUtils.exists(targetFile) then
                LrFileUtils.delete(targetFile)
            end
            LrFileUtils.copy(sourceFile, targetFile)
        end
    end
end

-- Run preset installation whenever plugin provider is evaluated
ensurePresetInstalled()

-- Hide sections of the export dialog that should be locked to 16-bit TIFF
exportServiceProvider.hideSections = {
    "exportLocation",
    "fileNaming",
    "fileSettings",
    "imageSize",
    "outputSharpening",
    "metadata",
    "watermarking",
    "video",
}

-- Strictly enforce 16-bit uncompressed TIFF, hidden scratch subfolder, and disable native export re-import
exportServiceProvider.updateExportSettings = function(exportSettings)
    exportSettings.LR_format = 'TIFF'
    exportSettings.LR_export_format = 'TIFF'
    exportSettings.LR_export_colorSpace = 'ProPhotoRGB'
    exportSettings.LR_export_bitDepth = '16'
    exportSettings.LR_export_tiff_compression = 'NONE'
    exportSettings.LR_export_destinationType = 'sourceFolder'
    exportSettings.LR_export_useSubfolder = true
    exportSettings.LR_export_subfolderName = '.focus_stack_tmp'
    exportSettings.LR_collisionHandling = 'rename'
    exportSettings.reimportExportedPhoto = false
    exportSettings.LR_reimportExportedPhoto = false
    exportSettings.LR_addToCatalog = false
    exportSettings.reimport_stackWithOriginal = false
end

-- Initialize default export dialog settings
exportServiceProvider.startDialog = function(propertyTable)
    ensurePresetInstalled()

    propertyTable.LR_format = 'TIFF'
    propertyTable.LR_export_format = 'TIFF'
    propertyTable.LR_export_colorSpace = 'ProPhotoRGB'
    propertyTable.LR_export_bitDepth = '16'
    propertyTable.LR_export_tiff_compression = 'NONE'
    propertyTable.LR_export_destinationType = 'sourceFolder'
    propertyTable.LR_export_useSubfolder = true
    propertyTable.LR_export_subfolderName = '.focus_stack_tmp'
    propertyTable.LR_collisionHandling = 'rename'
    propertyTable.reimportExportedPhoto = false
    propertyTable.LR_reimportExportedPhoto = false
    propertyTable.LR_addToCatalog = false
    propertyTable.reimport_stackWithOriginal = false

    if propertyTable.usePythonEngine == nil then
        propertyTable.usePythonEngine = true
    end

    if propertyTable.interactiveMode == nil then
        propertyTable.interactiveMode = true
    end

    if propertyTable.outputSuffix == nil then
        propertyTable.outputSuffix = "_stacked"
    end

    if propertyTable.stackAboveOriginal == nil then
        propertyTable.stackAboveOriginal = true
    end

    if propertyTable.deleteIntermediates == nil then
        propertyTable.deleteIntermediates = true
    end

    if propertyTable.batchSize == nil then
        propertyTable.batchSize = "8"
    end

    if propertyTable.fullResolutionAlign == nil then
        propertyTable.fullResolutionAlign = true
    end

    if propertyTable.globalAlign == nil then
        propertyTable.globalAlign = true
    end

    if propertyTable.preserveExposure == nil then
        propertyTable.preserveExposure = true
    end

    if propertyTable.bgMode == nil then
        propertyTable.bgMode = "auto"
    end
end

-- Custom UI section in Lightroom's Export dialog
exportServiceProvider.sectionsForTopOfDialog = function(f, propertyTable)
    local bind = LrView.bind
    local share = LrView.share

    return {
        {
            title = "Open Focus Stacking Settings",

            f:row {
                f:static_text {
                    title = "Stacking Engine:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:checkbox {
                    title = "16-Bit Pro Multi-Channel Engine (Zero Banding, ICC Preserved, No Halos)",
                    value = bind 'usePythonEngine',
                },
            },

            f:row {
                f:static_text {
                    title = "Interactivity:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:checkbox {
                    title = "Interactive Preview Mode (Opens Web UI with live 100% crop before render)",
                    value = bind 'interactiveMode',
                },
            },

            f:row {
                f:static_text {
                    title = "Background Mode:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:popup_menu {
                    value = bind 'bgMode',
                    items = {
                        { title = "Auto (Detects Polarity)", value = "auto" },
                        { title = "Dark (Black / Deep Bokeh)", value = "dark" },
                        { title = "Bright (White / High-Key)", value = "bright" },
                        { title = "Energy (Contrast / Foliage)", value = "energy" },
                    },
                    enabled = bind 'usePythonEngine',
                },
            },

            f:row {
                f:static_text {
                    title = "Output Suffix:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:edit_field {
                    value = bind 'outputSuffix',
                    width_in_chars = 15,
                },
                f:static_text {
                    title = " (e.g. photo123_stacked.tif)",
                    text_color = LrColor(0.5, 0.5, 0.5),
                },
            },

            f:row {
                f:static_text {
                    title = "Alignment Quality:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:checkbox {
                    title = "Full-resolution alignment (Legacy C++ engine only)",
                    value = bind 'fullResolutionAlign',
                    enabled = bind { key = 'usePythonEngine', transform = function(v) return not v end },
                },
            },

            f:row {
                f:static_text {
                    title = "",
                    width = share 'label_width',
                },
                f:checkbox {
                    title = "Global alignment (Legacy C++ engine only)",
                    value = bind 'globalAlign',
                    enabled = bind { key = 'usePythonEngine', transform = function(v) return not v end },
                },
            },

            f:row {
                f:static_text {
                    title = "Exposure & Color:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:checkbox {
                    title = "Preserve manual flash exposure (Legacy C++ engine only)",
                    value = bind 'preserveExposure',
                    enabled = bind { key = 'usePythonEngine', transform = function(v) return not v end },
                },
            },

            f:row {
                f:static_text {
                    title = "Batch Size:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:edit_field {
                    value = bind 'batchSize',
                    width_in_chars = 8,
                    enabled = bind { key = 'usePythonEngine', transform = function(v) return not v end },
                },
                f:static_text {
                    title = " (RAM saving setting; Legacy C++ engine only)",
                    text_color = LrColor(0.5, 0.5, 0.5),
                },
            },

            f:row {
                f:static_text {
                    title = "Workflow:",
                    alignment = 'right',
                    width = share 'label_width',
                },
                f:checkbox {
                    title = "Auto-stack merged image with originals (Leave unchecked for visible standalone thumbnail)",
                    value = bind 'stackAboveOriginal',
                },
            },

            f:row {
                f:static_text {
                    title = "",
                    width = share 'label_width',
                },
                f:checkbox {
                    title = "Delete temporary intermediate files after stacking",
                    value = bind 'deleteIntermediates',
                },
            },
        }
    }
end

-- Force Lightroom to demosaic RAW/DNG into pristine 16-bit TIFFs before passing to engine
exportServiceProvider.disallowFileFormats = { "JPEG", "PSD", "DNG", "ORIGINAL" }
exportServiceProvider.allowFileFormats = { "TIFF" }

exportServiceProvider.exportPresetChanges = {
    [ 'LR_format' ] = 'TIFF',
    [ 'LR_export_format' ] = 'TIFF',
    [ 'LR_export_colorSpace' ] = 'ProPhotoRGB',
    [ 'LR_export_bitDepth' ] = '16',
    [ 'LR_export_tiff_compression' ] = 'NONE',
    [ 'LR_export_destinationType' ] = 'sourceFolder',
    [ 'LR_export_useSubfolder' ] = true,
    [ 'LR_export_subfolderName' ] = '.focus_stack_tmp',
    [ 'reimportExportedPhoto' ] = false,
    [ 'LR_reimportExportedPhoto' ] = false,
    [ 'LR_addToCatalog' ] = false,
    [ 'reimport_stackWithOriginal' ] = false,
}

-- The actual execution pipeline
exportServiceProvider.processRenderedPhotos = function(functionContext, exportContext)
    local exportSession = exportContext.exportSession
    local exportParams = exportContext.propertyTable
    local nPhotos = exportSession:countRenditions()

    if nPhotos < 2 then
        LrDialogs.message("Focus Stacker", "Please select at least 2 bracketed photos to stack.", "warning")
        return
    end

    local progressScope = exportContext:configureProgress{
        title = "Focus Stacking " .. tostring(nPhotos) .. " photos...",
    }
    progressScope:setCaption("Rendering 16-bit TIFFs for focus stacking...")

    local renderedPaths = {}
    local pathToPhoto = {}
    local sourcePhotos = {}

    for i, rendition in exportContext:renditions{ stopIfCanceled = true } do
        if progressScope:isCanceled() then break end
        progressScope:setPortionComplete((i - 1) / (nPhotos * 2))

        local photo = rendition.photo
        if photo then
            table.insert(sourcePhotos, photo)
        end

        local success, path = rendition:waitForRender()
        if success then
            table.insert(renderedPaths, path)
            if photo then
                pathToPhoto[path] = photo
            end
        end
    end

    if progressScope:isCanceled() then
        return
    end

    if #renderedPaths < 2 then
        LrDialogs.message("Focus Stacker", "Failed to render photos.", "critical")
        return
    end

    -- Deterministically sort paths so the output is named after the first frame of the sequence
    table.sort(renderedPaths)
    local firstRenderedPath = renderedPaths[1]
    local renditionFolder = LrPathUtils.parent(firstRenderedPath)

    local catalog = LrApplication.activeCatalog()
    local targetPhotos = catalog:getTargetPhotos()
    local photoshootDir = nil
    local firstSourcePhoto = nil

    -- 1. Resolve directory from catalog:getTargetPhotos()
    if targetPhotos and #targetPhotos > 0 then
        for _, photo in ipairs(targetPhotos) do
            local rawPath = nil
            LrTasks.pcall(function()
                photo.catalog:withReadAccessDo(function()
                    rawPath = photo:getRawMetadata('path')
                end)
            end)
            if not rawPath then
                LrTasks.pcall(function()
                    rawPath = photo.path
                end)
            end
            if rawPath and #rawPath > 0 and not string.find(rawPath, "/var/folders", 1, true) and not string.find(rawPath, "/tmp", 1, true) and not string.find(rawPath, "/private/var", 1, true) then
                photoshootDir = LrPathUtils.parent(rawPath)
                firstSourcePhoto = photo
                break
            end
        end
    end

    -- 2. Resolve directory from rendition source photos if targetPhotos didn't provide it
    if not photoshootDir and #sourcePhotos > 0 then
        for _, photo in ipairs(sourcePhotos) do
            local rawPath = nil
            LrTasks.pcall(function()
                photo.catalog:withReadAccessDo(function()
                    rawPath = photo:getRawMetadata('path')
                end)
            end)
            if not rawPath then
                LrTasks.pcall(function()
                    rawPath = photo.path
                end)
            end
            if rawPath and #rawPath > 0 and not string.find(rawPath, "/var/folders", 1, true) and not string.find(rawPath, "/tmp", 1, true) and not string.find(rawPath, "/private/var", 1, true) then
                photoshootDir = LrPathUtils.parent(rawPath)
                firstSourcePhoto = photo
                break
            end
        end
    end

    -- 3. If renditionFolder is a user folder on disk (not OS temporary scratch), use it
    if not photoshootDir then
        if LrPathUtils.leafName(renditionFolder) == ".focus_stack_tmp" then
            local parentDir = LrPathUtils.parent(renditionFolder)
            if not string.find(parentDir, "/var/folders", 1, true) and not string.find(parentDir, "/tmp", 1, true) and not string.find(parentDir, "/private/var", 1, true) then
                photoshootDir = parentDir
            end
        elseif not string.find(renditionFolder, "/var/folders", 1, true) and not string.find(renditionFolder, "/tmp", 1, true) and not string.find(renditionFolder, "/private/var", 1, true) then
            photoshootDir = renditionFolder
        end
    end

    if not photoshootDir then
        progressScope:done()
        LrDialogs.message(
            "Focus Stacking Error",
            "Could not determine the photoshoot directory for the selected photos.\nNo output file was created.",
            "critical"
        )
        return
    end

    local exportFolder = photoshootDir
    if not firstSourcePhoto and targetPhotos and #targetPhotos > 0 then
        firstSourcePhoto = targetPhotos[1]
    end

    local isSubfolder = (LrPathUtils.leafName(renditionFolder) == ".focus_stack_tmp")
    local firstPhotoName = LrPathUtils.removeExtension(LrPathUtils.leafName(firstRenderedPath))

    progressScope:setPortionComplete(0.5)
    progressScope:setCaption("Normalizing orientation tags...")

    -- Fix TIFF orientation tag on camera raw exports (prevents OpenCV imread crash)
    local fixScript = getFixOrientationScriptPath()
    local pythonBin = getPythonPath()
    if fixScript and LrFileUtils.exists(fixScript) then
        local fixCmd = "\"" .. pythonBin .. "\" \"" .. fixScript .. "\""
        for _, path in ipairs(renderedPaths) do
            fixCmd = fixCmd .. " \"" .. path .. "\""
        end
        LrTasks.execute(fixCmd)
    end

    progressScope:setPortionComplete(0.55)
    progressScope:setCaption("Computing full-resolution focus stack on " .. #renderedPaths .. " frames (~1-2 min)...")

    local suffix = exportParams.outputSuffix or "_stacked"
    local firstExt = string.lower(LrPathUtils.extension(firstRenderedPath) or "")
    local outExt = (firstExt == "jpg" or firstExt == "jpeg") and ("." .. firstExt) or ".tif"
    LrFileUtils.createAllDirectories(exportFolder)
    local outputPath = LrPathUtils.child(exportFolder, firstPhotoName .. suffix .. outExt)
    local logPath = LrPathUtils.child(renditionFolder, "focus_stack.log")
    local scriptPath = LrPathUtils.child(renditionFolder, "focus_stack_run.sh")

    -- Recover any previous render accidentally placed into ~/Pictures
    local misplacedPath = LrPathUtils.child(LrPathUtils.getStandardFilePath('pictures'), firstPhotoName .. suffix .. ".tif")
    if exportFolder and misplacedPath ~= outputPath and LrFileUtils.exists(misplacedPath) and not LrFileUtils.exists(outputPath) then
        LrTasks.pcall(function() LrFileUtils.copy(misplacedPath, outputPath) end)
    end

    local venvPy, cliPy = getPythonEnginePaths()
    local usePython = (exportParams.usePythonEngine ~= false) and (venvPy ~= nil) and (cliPy ~= nil)

    if usePython then
        local isInteractive = (exportParams.interactiveMode ~= false)

        progressScope:setPortionComplete(0.55)
        if isInteractive then
            progressScope:setCaption("Opening Interactive Preview window...")
        else
            progressScope:setCaption("Computing 16-bit multi-channel stack on " .. #renderedPaths .. " frames (~30-60s)...")
        end

        local flags = { "--output=\"" .. outputPath .. "\"" }
        if isInteractive then
            table.insert(flags, "--interactive")
        end
        if exportParams.bgMode then
            table.insert(flags, "--bg-mode=" .. tostring(exportParams.bgMode))
        end
        table.insert(flags, "--crop")
        table.insert(flags, "--lightroom")

        -- Create temporary script inside the scratch folder
        local scriptFile, err = io.open(scriptPath, "w")
        if not scriptFile then
            LrDialogs.message("Error", "Could not create execution script: " .. tostring(err), "critical")
            return
        end

        scriptFile:write("#!/bin/bash\n")
        scriptFile:write("export FOCUS_STACK_LIGHTROOM=1\n")
        scriptFile:write("\"" .. venvPy .. "\" \"" .. cliPy .. "\"")
        for _, flag in ipairs(flags) do
            scriptFile:write(" " .. flag)
        end
        for _, path in ipairs(renderedPaths) do
            scriptFile:write(" \"" .. path .. "\"")
        end
        scriptFile:write(" > \"" .. logPath .. "\" 2>&1\n")
        scriptFile:close()

        LrTasks.execute("/bin/chmod +x \"" .. scriptPath .. "\"")
        local exitCode = LrTasks.execute("\"" .. scriptPath .. "\"")
        LrFileUtils.delete(scriptPath)

        if not LrFileUtils.exists(outputPath) then
            local stem = LrPathUtils.removeExtension(outputPath)
            local altTif = stem .. ".tif"
            local altJpg = stem .. ".jpg"
            if LrFileUtils.exists(altJpg) then
                outputPath = altJpg
            elseif LrFileUtils.exists(altTif) then
                outputPath = altTif
            end
        end

        if exitCode ~= 0 or not LrFileUtils.exists(outputPath) then
            if exitCode == 2 or exitCode == 512 then
                -- Stacking was cancelled by the user in the interactive preview UI
                if LrFileUtils.exists(logPath) then
                    LrFileUtils.delete(logPath)
                end
                if exportParams.deleteIntermediates ~= false then
                    if isSubfolder then
                        LrFileUtils.delete(renditionFolder)
                    else
                        for _, path in ipairs(renderedPaths) do
                            LrFileUtils.delete(path)
                        end
                    end
                end
                return
            end

            local logContent = ""
            if LrFileUtils.exists(logPath) then
                local f = io.open(logPath, "r")
                if f then
                    logContent = f:read("*a")
                    f:close()
                end
            end
            if #logContent > 1000 then
                logContent = string.sub(logContent, #logContent - 1000)
            end
            LrDialogs.message(
                "Focus Stacking Failed",
                "16-bit stacking process exited with status code " .. tostring(exitCode) .. ".\n\nDetails:\n" .. logContent,
                "critical"
            )
            return
        end
    else
        -- Fallback to legacy focus-stack C++ binary
        local binary = getBinaryPath()
        if not binary then
            LrDialogs.message(
                "focus-stack Not Found",
                "Could not locate the focus-stack binary inside the plugin bundle.\nPlease ensure FocusStack.lrplugin/bin is intact.",
                "critical"
            )
            return
        end

        local batchSize = exportParams.batchSize or "8"
        local flags = { "--no-opencl" }

        if exportParams.fullResolutionAlign ~= false then
            table.insert(flags, "--full-resolution-align")
        end

        if exportParams.globalAlign ~= false then
            table.insert(flags, "--global-align")
        end

        if exportParams.preserveExposure ~= false then
            table.insert(flags, "--no-contrast")
            table.insert(flags, "--no-whitebalance")
        end

        table.insert(flags, "--batchsize=" .. tostring(batchSize))
        table.insert(flags, "--output=\"" .. outputPath .. "\"")

        local scriptFile, err = io.open(scriptPath, "w")
        if not scriptFile then
            LrDialogs.message("Error", "Could not create execution script: " .. tostring(err), "critical")
            return
        end

        scriptFile:write("#!/bin/bash\n")
        scriptFile:write("\"" .. binary .. "\"")
        for _, flag in ipairs(flags) do
            scriptFile:write(" " .. flag)
        end
        for _, path in ipairs(renderedPaths) do
            scriptFile:write(" \"" .. path .. "\"")
        end
        scriptFile:write(" > \"" .. logPath .. "\" 2>&1\n")
        scriptFile:close()

        LrTasks.execute("/bin/chmod +x \"" .. scriptPath .. "\"")
        local exitCode = LrTasks.execute("\"" .. scriptPath .. "\"")
        LrFileUtils.delete(scriptPath)

        if exitCode ~= 0 or not LrFileUtils.exists(outputPath) then
            local logContent = ""
            if LrFileUtils.exists(logPath) then
                local f = io.open(logPath, "r")
                if f then
                    logContent = f:read("*a")
                    f:close()
                end
            end
            if #logContent > 1000 then
                logContent = string.sub(logContent, #logContent - 1000)
            end
            LrDialogs.message(
                "Focus Stacking Failed",
                "focus-stack process exited with status code " .. tostring(exitCode) .. ".\n\nDetails:\n" .. logContent,
                "critical"
            )
        end
    end

    -- Clean up log file on success
    if LrFileUtils.exists(logPath) then
        LrFileUtils.delete(logPath)
    end

    -- Clean up intermediate files
    local stdPath = LrPathUtils.standardizePath(outputPath)
    if exportParams.deleteIntermediates ~= false then
        progressScope:setCaption("Cleaning up temporary scratch files...")
        local isOutputInRendition = string.find(stdPath, renditionFolder, 1, true) ~= nil
        if isSubfolder and not isOutputInRendition and renditionFolder ~= exportFolder then
            -- Delete the hidden .focus_stack_tmp directory entirely
            LrFileUtils.delete(renditionFolder)
        else
            for _, path in ipairs(renderedPaths) do
                local sPath = LrPathUtils.standardizePath(path)
                if sPath ~= stdPath then
                    LrFileUtils.delete(path)
                end
            end
        end
    end

    -- Re-import the stacked result back into the Lightroom Catalog
    progressScope:setPortionComplete(0.95)
    progressScope:setCaption("Importing stacked photo into Lightroom catalog...")

    -- Ensure filesystem writes are fully committed and disk handles settle
    LrTasks.sleep(1.0)

    local function debugLog(msg)
        local f = io.open("/tmp/focus_stack_import.log", "a")
        if f then
            f:write(os.date("%Y-%m-%d %H:%M:%S") .. " [Build 21] " .. tostring(msg) .. "\n")
            f:close()
        end
    end

    debugLog("==========================================")
    debugLog("Starting catalog import sequence")
    debugLog("outputPath: " .. tostring(outputPath))
    debugLog("stdPath: " .. tostring(stdPath))
    debugLog("File exists on disk: " .. tostring(LrFileUtils.exists(stdPath)))

    if not LrFileUtils.exists(stdPath) then
        progressScope:done()
        LrDialogs.message(
            "Focus Stacking Error",
            "The expected stacked output file could not be found on disk:\n" .. stdPath,
            "critical"
        )
        return
    end

    -- If invoked from FocusStackMenuAction, skip in-session catalog ingestion.
    -- MenuAction will perform uncontested catalog ingestion immediately after
    -- doExportOnCurrentTask() completes.
    if exportParams.invokedFromMenuAction == true then
        progressScope:setPortionComplete(1.0)
        progressScope:done()
        debugLog("invokedFromMenuAction is true: skipping in-session ingestion so FocusStackMenuAction performs uncontested post-export ingestion.")
        return
    end

    local catalog = LrApplication.activeCatalog()

    local stackWith = nil
    local position = nil
    if exportParams.stackAboveOriginal == true and firstSourcePhoto then
        stackWith = firstSourcePhoto
        position = "above"
    end
    debugLog("stackWith: " .. tostring(stackWith) .. ", position: " .. tostring(position))

    local function tryImportPhoto()
        local imported = nil
        local importErr = nil

        -- 1. Check if already indexed
        catalog:withReadAccessDo(function()
            imported = catalog:findPhotoByPath(stdPath)
        end)
        if imported then
            debugLog("tryImportPhoto: Photo was ALREADY found in catalog via findPhotoByPath")
            return imported, nil
        end

        -- 2. Try write access
        debugLog("tryImportPhoto: catalog.hasWriteAccess=" .. tostring(catalog.hasWriteAccess))
        if catalog.hasWriteAccess then
            local addSuccess = false
            local stackErrMsg = nil
            local plainErrMsg = nil

            if stackWith then
                local stackOk, stackRes, stackMsg = LrTasks.pcall(function()
                    return catalog:addPhoto(stdPath, stackWith, position)
                end)
                debugLog("tryImportPhoto (direct): stack attempt -> ok=" .. tostring(stackOk) .. ", res=" .. tostring(stackRes) .. ", msg=" .. tostring(stackMsg))
                if stackOk and stackRes then
                    addSuccess = true
                else
                    stackErrMsg = stackMsg or (not stackOk and tostring(stackRes)) or "stacking failed"
                end
            end

            if not addSuccess then
                local plainOk, plainRes, plainMsg = LrTasks.pcall(function()
                    return catalog:addPhoto(stdPath)
                end)
                debugLog("tryImportPhoto (direct): standalone attempt -> ok=" .. tostring(plainOk) .. ", res=" .. tostring(plainRes) .. ", msg=" .. tostring(plainMsg))
                if plainOk and plainRes then
                    addSuccess = true
                else
                    plainErrMsg = plainMsg or (not plainOk and tostring(plainRes)) or "standalone addPhoto returned false"
                end
            end

            if not addSuccess then
                local errParts = {}
                if stackErrMsg then table.insert(errParts, "Stack: " .. tostring(stackErrMsg)) end
                if plainErrMsg then table.insert(errParts, "Standalone: " .. tostring(plainErrMsg)) end
                importErr = table.concat(errParts, " | ")
            end
        else
            local withWriteStatus, withWriteMsg = catalog:withWriteAccessDo("Import Stacked Photo", function()
                local addSuccess = false
                local stackErrMsg = nil
                local plainErrMsg = nil

                if stackWith then
                    local stackOk, stackRes, stackMsg = LrTasks.pcall(function()
                        return catalog:addPhoto(stdPath, stackWith, position)
                    end)
                    debugLog("tryImportPhoto: stack attempt -> ok=" .. tostring(stackOk) .. ", res=" .. tostring(stackRes) .. ", msg=" .. tostring(stackMsg))
                    if stackOk and stackRes then
                        addSuccess = true
                    else
                        stackErrMsg = stackMsg or (not stackOk and tostring(stackRes)) or "stacking failed"
                    end
                end

                if not addSuccess then
                    local plainOk, plainRes, plainMsg = LrTasks.pcall(function()
                        return catalog:addPhoto(stdPath)
                    end)
                    debugLog("tryImportPhoto: standalone attempt -> ok=" .. tostring(plainOk) .. ", res=" .. tostring(plainRes) .. ", msg=" .. tostring(plainMsg))
                    if plainOk and plainRes then
                        addSuccess = true
                    else
                        plainErrMsg = plainMsg or (not plainOk and tostring(plainRes)) or "standalone addPhoto returned false"
                    end
                end

                if not addSuccess then
                    local errParts = {}
                    if stackErrMsg then table.insert(errParts, "Stack: " .. tostring(stackErrMsg)) end
                    if plainErrMsg then table.insert(errParts, "Standalone: " .. tostring(plainErrMsg)) end
                    importErr = table.concat(errParts, " | ")
                end
            end, { timeout = 25 })

            debugLog("tryImportPhoto: withWriteAccessDo returned withWriteStatus=" .. tostring(withWriteStatus) .. ", withWriteMsg=" .. tostring(withWriteMsg))

            if withWriteStatus ~= "executed" and withWriteStatus ~= "queued" then
                importErr = "Catalog write " .. tostring(withWriteStatus) .. ": " .. tostring(withWriteMsg or "lock busy")
            end
        end

        -- 3. Always retrieve actual LrPhoto instance via findPhotoByPath
        catalog:withReadAccessDo(function()
            imported = catalog:findPhotoByPath(stdPath)
        end)

        if imported then
            debugLog("tryImportPhoto: Photo found in catalog via findPhotoByPath!")
            importErr = nil
        end

        return imported, importErr
    end

    local function finalizeSelection(photo)
        if not photo then return end
        debugLog("finalizeSelection: adding to collections and selecting in grid")
        if firstSourcePhoto then
            LrTasks.pcall(function()
                local collections = nil
                catalog:withReadAccessDo(function()
                    collections = firstSourcePhoto:getContainedCollections()
                end)
                if collections and #collections > 0 then
                    catalog:withWriteAccessDo("Add Stacked Photo to Collections", function()
                        for _, col in ipairs(collections) do
                            col:addPhotos({ photo })
                        end
                    end, { timeout = 10 })
                end
            end)
        end

        LrTasks.pcall(function()
            local folder = nil
            catalog:withReadAccessDo(function()
                folder = catalog:getFolderByPath(exportFolder)
                if not folder then
                    folder = catalog:getFolderByPath(exportFolder:gsub("[/\\]+$", ""))
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
            catalog:setSelectedPhotos(photo, {})
        end)
    end

    -- Close the export progress scope immediately so Lightroom can finalize
    -- the export session and release all export locks
    progressScope:setPortionComplete(1.0)
    progressScope:done()

    debugLog("Spawning detached post-export ingestion task...")
    LrTasks.startAsyncTask(function()
        -- Wait for export session to completely release all locks
        LrTasks.sleep(0.8)
        local postPhoto, postErr = nil, nil
        for postAttempt = 1, 6 do
            debugLog("Post-export attempt " .. postAttempt)
            postPhoto, postErr = tryImportPhoto()
            if postPhoto then
                finalizeSelection(postPhoto)
                break
            end
            LrTasks.sleep(0.5)
        end

        debugLog("Post-export task result: postPhoto=" .. tostring(postPhoto) .. ", postErr=" .. tostring(postErr))

        if postPhoto then
            debugLog("Focus Stacking Complete: successfully stacked " .. tostring(#renderedPaths) .. " photos, imported: " .. tostring(stdPath))
        else
            LrDialogs.message(
                "Focus Stacking Import Diagnostics",
                "The 16-bit stacked photo was saved to disk:\n" .. stdPath .. "\n\n" ..
                "Lightroom catalog registration diagnostics (Build 21):\n" ..
                "Post-export error: " .. tostring(postErr or "photo not returned by catalog") .. "\n\n" ..
                "Full log written to: /tmp/focus_stack_import.log",
                "critical"
            )
        end
    end)
end

return exportServiceProvider
