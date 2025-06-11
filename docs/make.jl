using Documenter
using L2ODLL


include("definitions.jl")

makedocs(
    modules=[L2ODLL],
    sitename = "L2ODLL.jl",
    format = Documenter.HTML(;
        assets = ["assets/wider.css", "assets/redlinks.css"],
        mathengine = Documenter.MathJax3(Dict(
            :tex => Dict(
                "macros" => make_macros_dict("docs/src/assets/definitions.tex"),
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
            ),
        )),
    ),
    pages = [
        "Home" => "index.md",
        "Background" => [
            "background/decompositions.md",
            "background/decomposition_detection.md",
        ],
        "Reference" => "public.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/LearningToOptimize/L2ODLL.jl.git",
    push_preview=true,
)
