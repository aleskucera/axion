window.MathJax = {
	tex: {
		inlineMath: [["\\(", "\\)"]], // For inline math like \( E = mc^2 \)
		displayMath: [["\\[", "\\]"]], // For display math like \[ ... \]
		processEscapes: true,
		processEnvironments: true,
	},
	options: {
		ignoreHtmlClass: ".*|",
		processHtmlClass: "arithmatex",
	},
};
