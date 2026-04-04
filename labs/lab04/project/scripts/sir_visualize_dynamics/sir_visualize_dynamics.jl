using DrWatson
@quickactivate "project"
using Agents, DataFrames, Plots, CSV

include(srcdir("sir_model.jl"))

df = CSV.read(datadir("beta_scan_all.csv"), DataFrame)

p1 = plot(
    df.beta,
    df.peak,
    label = "Пик",
    xlabel = "β",
    ylabel = "Доля инфицированных",
    linewidth = 2,
    marker = :circle,
)

plot!(
    p1,
    df.beta,
    df.final_inf,
    label = "Конечная",
    linewidth = 2,
    marker = :square,
)

p2 = plot(
    df.beta,
    df.deaths,
    xlabel = "β",
    ylabel = "Число умерших",
    linewidth = 2,
    marker = :diamond,
    legend = false,
)

p3 = plot(
    df.beta,
    df.final_rec,
    xlabel = "β",
    ylabel = "Доля выздоровевших",
    linewidth = 2,
    marker = :circle,
    legend = false,
)

plot(p1, p2, p3, layout = (3, 1), size = (800, 900))

savefig(plotsdir("comprehensive_analysis.png"))

println("Комплексный график сохранён в plots/comprehensive_analysis.png")
