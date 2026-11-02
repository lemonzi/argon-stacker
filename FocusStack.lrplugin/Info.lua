return {
    LrSdkVersion = 11.0,
    LrSdkMinimumVersion = 6.0,
    LrToolkitIdentifier = 'com.lemonzi.argonstacker',
    LrPluginName = 'Argon Stacker',
    LrPluginInfoUrl = 'https://github.com/lemonzi/argon-stacker',

    LrExportServiceProvider = {
        title = "Argon Stacker",
        file = "FocusStackExportServiceProvider.lua",
    },

    LrExportMenuItems = {
        {
            title = "Focus Stack with Argon Stacker",
            file = "FocusStackMenuAction.lua",
        },
    },

    LrLibraryMenuItems = {
        {
            title = "Focus Stack with Argon Stacker",
            file = "FocusStackMenuAction.lua",
        },
    },

    VERSION = { major = 1, minor = 0, revision = 0, build = 44 },
}
