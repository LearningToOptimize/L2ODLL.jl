using Documenter
using Literate
using L2ODLL
using L2OCore: L2OCore  # loads the extension, so its docstrings are included

include("definitions.jl")

# Literate examples: docs/literate/*.jl -> docs/src/examples/*.md
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const EXAMPLES_DIR = joinpath(@__DIR__, "src", "examples")
for file in readdir(LITERATE_DIR)
    endswith(file, ".jl") || continue
    Literate.markdown(joinpath(LITERATE_DIR, file), EXAMPLES_DIR; documenter=true, credit=false)
end

const EXT = Base.get_extension(L2ODLL, :L2ODLLL2OCoreExt)

DocMeta.setdocmeta!(L2ODLL, :DocTestSetup, :(using L2ODLL); recursive=true)

makedocs(;
    modules=[L2ODLL, EXT],
    sitename="L2ODLL.jl",
    authors="LearningToOptimize contributors",
    doctest=true,
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://learningtooptimize.github.io/L2ODLL.jl",
        assets=["assets/wider.css", "assets/redlinks.css", "assets/figures.css"],
        mathengine=Documenter.MathJax3(
            Dict(
                :tex => Dict(
                    "macros" => make_macros_dict(
                        joinpath(@__DIR__, "src", "assets", "definitions.tex")
                    ),
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                ),
            ),
        ),
    ),
    pages=[
        "Home" => "index.md",
        "Theory" => [
            "Dual Lagrangian Learning" => "theory.md",
            "background/decompositions.md",
            "background/decomposition_detection.md",
        ],
        "Package guide" => [
            "Getting started and extension points" => "guide/concepts.md",
            "L2OCore interface and design notes" => "guide/l2ocore.md",
        ],
        "Examples" => [
            "Markowitz portfolio (SOCP)" => "examples/markowitz.md",
            "Economic dispatch: training a proxy" => "examples/economic_dispatch.md",
        ],
        "API reference" => "public.md",
    ],
)

deploydocs(;
    repo="github.com/LearningToOptimize/L2ODLL.jl.git", devbranch="main", push_preview=true
)
