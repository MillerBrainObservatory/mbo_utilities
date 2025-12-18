' MBO Utilities Windows Launcher - runs without console window

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

Function GetEnvOrDefault(envVar, defaultVal)
    Dim val
    val = WshShell.ExpandEnvironmentStrings("%" & envVar & "%")
    If val = "%" & envVar & "%" Or val = "" Then
        GetEnvOrDefault = defaultVal
    Else
        GetEnvOrDefault = val
    End If
End Function

Function FindMboExe()
    Dim paths, p, userProfile, xdgDataHome, localAppData

    userProfile = WshShell.ExpandEnvironmentStrings("%USERPROFILE%")
    localAppData = WshShell.ExpandEnvironmentStrings("%LOCALAPPDATA%")

    ' uv uses XDG_DATA_HOME on windows, defaults to LOCALAPPDATA
    xdgDataHome = GetEnvOrDefault("XDG_DATA_HOME", localAppData)

    ' check UV_TOOL_DIR env var first
    Dim uvToolDir
    uvToolDir = GetEnvOrDefault("UV_TOOL_DIR", "")

    Dim appDataRoaming
    appDataRoaming = WshShell.ExpandEnvironmentStrings("%APPDATA%")

    ' build search paths in priority order
    paths = Array( _
        uvToolDir & "\mbo-utilities\Scripts\mbo.exe", _
        xdgDataHome & "\uv\tools\mbo-utilities\Scripts\mbo.exe", _
        appDataRoaming & "\uv\tools\mbo-utilities\Scripts\mbo.exe", _
        localAppData & "\uv\tools\mbo-utilities\Scripts\mbo.exe", _
        userProfile & "\.local\share\uv\tools\mbo-utilities\Scripts\mbo.exe", _
        userProfile & "\.local\bin\mbo.exe" _
    )

    For Each p In paths
        If p <> "\mbo-utilities\Scripts\mbo.exe" And fso.FileExists(p) Then
            FindMboExe = p
            Exit Function
        End If
    Next

    FindMboExe = ""
End Function

Dim mboExe
mboExe = FindMboExe()

If mboExe <> "" Then
    ' run hidden (0) - no console window
    WshShell.Run """" & mboExe & """", 0, False
Else
    MsgBox "Could not find mbo.exe. Please reinstall mbo_utilities." & vbCrLf & vbCrLf & _
           "Expected location: %LOCALAPPDATA%\uv\tools\mbo-utilities\Scripts\mbo.exe", _
           vbExclamation, "MBO Utilities"
End If
