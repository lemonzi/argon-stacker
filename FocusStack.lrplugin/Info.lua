return {
    LrSdkVersion = 11.0,
    LrSdkMinimumVersion = 6.0,
    LrToolkitIdentifier = 'com.antigravity.openfocusstack',
    LrPluginName = 'Open Focus Stack',
    LrPluginInfoUrl = 'https://github.com/PetteriAimonen/focus-stack',

    LrExportServiceProvider = {
        title = "Open Focus Stack",
        file = "FocusStackExportServiceProvider.lua",
    },

    LrExportMenuItems = {
        {
            title = "Focus Stack with Open Focus Stack",
            file = "FocusStackMenuAction.lua",
        },
    },

    LrLibraryMenuItems = {
        {
            title = "Focus Stack with Open Focus Stack",
            file = "FocusStackMenuAction.lua",
        },
    },

    VERSION = { major = 1, minor = 0, revision = 0, build = 39 },
}
