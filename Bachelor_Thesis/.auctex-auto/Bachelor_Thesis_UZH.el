(TeX-add-style-hook
 "Bachelor_Thesis_UZH"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8") ("fontenc" "T1") ("color" "usenames" "dvipsnames") ("geometry" "margin=1in" "includefoot") ("url" "hyphens") ("siunitx" "separate-uncertainty=true" "multi-part-units=single" "detect-weight=true" "detect-family=true") ("tcolorbox" "many") ("hyperref" "colorlinks=true" "linkcolor=blue")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "babel"
    "inputenc"
    "fontenc"
    "graphics"
    "color"
    "graphicx"
    "dcolumn"
    "bm"
    "epsfig"
    "pdflscape"
    "multirow"
    "rotating"
    "longtable"
    "lineno"
    "geometry"
    "fancyhdr"
    "url"
    "subcaption"
    "booktabs"
    "titlesec"
    "biblatex"
    "float"
    "textcomp"
    "sidecap"
    "wrapfig"
    "siunitx"
    "physics"
    "enumitem"
    "ulem"
    "mhchem"
    "pdfpages"
    "amsmath"
    "amsthm"
    "amssymb"
    "amsfonts"
    "mathtools"
    "centernot"
    "gensymb"
    "bbm"
    "mathrsfs"
    "tcolorbox"
    "empheq"
    "microtype"
    "hyperref")
   (TeX-add-symbols
    '("rom" 1)
    '("Lim" 1)
    "myeq"
    "R"
    "C"
    "Z"
    "N"
    "Q"
    "vc"
    "arcsinh"
    "supp"
    "gram"
    "rot"
    "divt"
    "sgn")
   (LaTeX-add-labels
    "eq:Energy_Density_Matrix"
    "eq:secondterm2"
    "eq:Kommutatorrelation_umgeformt"
    "eq:first_term_correlation_matrix"
    "eq:second_term_correlation_matrix"
    "eq:Correlation_Matrix_singlequench"
    "eq:expression_b_j"
    "eq:expression_left_Evolution")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

