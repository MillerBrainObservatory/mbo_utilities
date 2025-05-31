import sys


def test_imgui_context_creation():
    if sys.platform == "win32":
        return

    # Check that this complex issue is fixed:
    #     https://github.com/pthom/imgui_bundle/issues/170#issuecomment-1900100904
    from imgui_bundle import imgui

    ctx = imgui.create_context()
    assert ctx is not None
    imgui.destroy_context(ctx)
