' MBO Utilities Launcher - runs mbo.exe without showing console window
' This script is created by install.ps1 and used by the desktop shortcut

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' find mbo.exe in common locations
mboExe = ""

' try uv tools location first
uvToolsPath = WshShell.ExpandEnvironmentStrings("%LOCALAPPDATA%\uv\tools\mbo_utilities\Scripts\mbo.exe")
If fso.FileExists(uvToolsPath) Then
    mboExe = uvToolsPath
End If

' try .local\bin
If mboExe = "" Then
    localBinPath = WshShell.ExpandEnvironmentStrings("%USERPROFILE%\.local\bin\mbo.exe")
    If fso.FileExists(localBinPath) Then
        mboExe = localBinPath
    End If
End If

' run mbo.exe hidden (0 = hidden window) with --splash flag for loading indicator
If mboExe <> "" Then
    WshShell.Run """" & mboExe & """ --splash", 0, False
Else
    ' fallback: show error
    MsgBox "Could not find mbo.exe. Please reinstall mbo_utilities.", vbExclamation, "MBO Utilities"
End If
