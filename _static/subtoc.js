// Make the sub-toctree elements a slightly smaller font size

var l1FontSize = window.getComputedStyle(
  document.querySelector(".toctree-l1"),
).fontSize;

var l2f;

var l1FontSizeNumber = parseFloat(l1FontSize);

document.querySelector(".toctree-l2").style.fontSize =
  l1FontSizeNumber - 1 + "rem";
