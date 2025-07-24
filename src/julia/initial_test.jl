#%% Julia setup
import BeforeIT as Bit
import Plots, StatsPlots

#%% Load params and data
parameters = Bit.AUSTRIA2010Q1.parameters
initial_conditions = Bit.AUSTRIA2010Q1.initial_conditions

#%% Run model
T = 20
model = Bit.Model(parameters, initial_conditions)

for t in 1:T
    Bit.step!(model; multi_threading = true)
    Bit.update_data!(model)
    print("step = ", t, "\n")
end

#%% Charts
Plots.plot(model.data.real_gdp, title = "gdp", titlefont = 10)
ps = Bit.plot_data(model, quantities = [:real_gdp, :real_household_consumption, :real_government_consumption, :real_capitalformation, :real_exports, :real_imports, :wages, :euribor, :gdp_deflator])
Plots.plot(ps..., layout = (3, 3))

#%% Run an ensemble for Monte Carlo sampling
model = Bit.Model(parameters, initial_conditions)
model_vec = Bit.ensemblerun(model, T, 4)

#%% Charts, using the vector of outputs
ps = Bit.plot_data_vector(model_vec)
Plots.plot(ps..., layout = (3, 3))

#%% Test the model objects
fieldnames(typeof(model.data))