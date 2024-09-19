"""
MIT License

Copyright (c) 2024 Diego Dorn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from contextlib import contextmanager

import streamlit as st

__all__ = ["st_horizontal"]

HORIZONTAL_STYLE = """
<style class="hide-element">
    /* Hides the style container and removes the extra spacing */
    .element-container:has(.hide-element) {
        display: none;
    }
    /*
        The selector for >.element-container is necessary to avoid selecting the whole
        body of the streamlit app, which is also a stVerticalBlock.
    */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) {
        display: flex;
        flex-direction: row !important;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: baseline;
        justify-content: end;
    }
    /* Buttons and their parent container all have a width of 704px, which we need to override */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div {
        width: max-content !important;
    }

    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) button {
        padding: 0.2rem 0.5rem;
        min-height: 0;
        background: rgba(0, 0, 0, 0.00);
    }
</style>
"""


def write_style():
    st.markdown(HORIZONTAL_STYLE, unsafe_allow_html=True)


@contextmanager
def st_horizontal():
    write_style()

    with st.container():
        st.markdown('<span class="hide-element horizontal-marker"></span>', unsafe_allow_html=True)
        yield


if __name__ == "__main__":
    buttons = [
        "Allow",
        "Deny",
        "Always Allow",
        "Edit",
        "More Options",
    ] * 2

    st.header("With the new horizontal layout")
    st.button("✅ Yes", key="okded")

    with st_horizontal():
        st.write("Confirm?")
        st.button("✅ Yes")
        st.button("❌ No")

    with st_horizontal():
        for i, option in enumerate(buttons):
            st.button(option, key=f"button_{i}")
    st.header("With columns")

    cols = st.columns(len(buttons))
    for i, option in enumerate(buttons):
        cols[i].button(option, key=f"button_col_{i}")

    st.header("Sample elements to check that we did not break anything")

    st.button("A button")
    st.button("Another button")
    with st.expander("Code"):
        st.code(
            """
    print("Hello, world!")
        """,
            language="python",
        )

    cols = st.columns(3)
    for i, col in enumerate(cols):
        col.write(f"Column {i}")
        col.button("Click me", key=f"col_{i}")

    with st.container(border=True):
        st.write("Inside container")
        st.button("Click me", key="container")
        st.button("Click me", key="container1")
