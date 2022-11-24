#This is unused code that may be fixed or used later don't run

"""
# Import pandas
import pandas as pd
 
# Read the file into a DataFrame: df
df = pd.read_csv('Housing_Prices.csv')
 
# Print the head of df
print(df.head())
 
# Print the tail of df
print(df.tail())
 
# Print the shape of df
print(df.shape)
 
# Print the columns of df
print(df.columns)
 
 
print(df.info())
 
 
# Calculating summary statistics
print(df.describe())


#Histogram  
HistDublin = df["Dublin"]
HistDublin.plot(y = "Year", kind = "hist")

HistCork = df["Cork"]
HistCork.plot(y = "Year", kind = "hist")

HistGalway = df["Galway"]
HistGalway.plot(y = "Year", kind = "hist")

HistLimerick = df["Limerick"]
HistLimerick.plot(y = "Year", kind = "hist")

HistWaterford = df["Waterford"]
HistWaterford.plot(y = "Year", kind = "hist")


#kdeplot
plt.figure(figsize=(19,6))
sns.kdeplot(df["Dublin"][df['Year'] == 1997], color="darkturquoise", shade=True)
sns.kdeplot(df["Dublin"][df['Year'] == 2015], color="lightcoral", shade=True)
plt.legend(['Year', 'County'])
plt.title('Average House price of Dublin 1997 and 2015')
plt.show()


#dub_1997 = [df['Year'] == 2015]
#dub_2015 = df.iloc[18]
#print(dub_1997)
#print(dub_2015)

Dublin = df['Dublin']

cdf_dub = Cdf(Dublin)

print(cdf_dub[5])

dub_price = df.Dublin

cdf_dub.plot()

plt.xlabel('Year')
plt.ylabel('CDF')
plt.show()

# Extract realinc and compute its log
prices = df['Dublin']
log_price = np.log10(prices)

# Compute mean and standard deviation
mean = log_price.mean()
std = log_price.std()
print(mean, std)

# Make a norm object
from scipy.stats import norm
dist = norm(mean, std)


# Evaluate the model CDF
xs = np.linspace(2, 5.5)
ys = dist.cdf(xs)

# Plot the model CDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Create and plot the Cdf of log_income
Cdf(log_price).plot() 
# Label the axes
plt.xlabel('log10 of Dublin')
plt.ylabel('CDF')
plt.show()



# Generate many Bootstrap replicates
def draw_bs_reps(data, func, size=1):
    #Draw bootstrap replicates.

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates
"""