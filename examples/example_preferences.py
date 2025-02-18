# preferences_example.py

from microeconomics.preferences import UtilityCurve
from plotting.plotter import Plotter

# Define the utility functions
utility1 = UtilityCurve(curve_type='cobb-douglas', alpha=0.3, beta=1.7, income=100, price_x=1, price_y=1)
utility2 = UtilityCurve(curve_type='cobb-douglas', alpha=0.7, beta=0.3, income=100, price_x=1, price_y=1)
utility3 = UtilityCurve(curve_type='cobb-douglas', alpha=1, beta=2, income=100, price_x=1, price_y=1)
utility4 = UtilityCurve(curve_type='cobb-douglas', alpha=1, beta=1, income=100, price_x=1, price_y=1)

### Calculation capabilities
print(f'Utility expression: {utility1.utility_expr}')
print(f'Numeric expression: {utility1.numeric_expr}')
print(f'Utility: {utility1.utility}')
print(f'Marshallian demand: {utility1.marshallian_demand}')
print(f'Hicksian demand: {utility1.calculate_hicksian_demand(utility1.utility, 2, 2)}')
print(f'Slutsky substitution (Total/ Income/ Substitution): {utility1.slutsky_substitution(new_price_x=0.5)}')
print(f'Income elasticity: {utility1.income_elasticity_x:.2g}')
print(f'Price elasticity: {utility1.price_elasticity_x:.2g}')
print(f'Cross price elasticity: {utility1.cross_price_elasticity}')
print(f'Good type: {utility1.classification_x}')


### Plotting capabilities

# List of plot objects
plots = [utility1, utility2, utility3, utility4]

# Define subtitles for each subplot
subtitles = []
# Create the Plotter instance
plotter = Plotter(rows=2, cols=2, plots=plots, title='Preferences Analysis', cmap='plasma', label=True,
                  label_position='first', label_direction='horizontal', show_colorbar=True, show_equation=True,
                  show_prices=True, show_optima=True, show_restriction=True, subtitles=subtitles,
                  box_position=[('low','right'), ('low','right'), ('low','right'), ('low','left')])

# Display the plots
plotter.show()