import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# headings
title = "Dating App Simulation"
st.title(title)
st.write("by [Harsh](https://www.linkedin.com/in/harshjain0016/)")

st.sidebar.title("Parameters")

# user inputs on sidebar ---------------------------------
population = st.sidebar.slider(
    "Population", value=100000, min_value=10000, max_value=500000, key=1
)

men_percent = st.sidebar.slider(
    "Percentage of men", value=75, min_value=1, max_value=99, key=2
)
women_percent = 100 - men_percent

matches_per_day_men = st.sidebar.slider(
    "Maximum number of matches per day (men)",
    value=20,
    min_value=1,
    max_value=50,
    key=3,
)
matches_per_day_women = st.sidebar.slider(
    "Maximum number of matches per day (women)",
    value=20,
    min_value=1,
    max_value=50,
    key=4,
)
number_of_days = st.sidebar.slider(
    "Number of days to run the simulation for",
    value=10,
    min_value=1,
    max_value=20,
    key=5,
)
matching_criteria = st.sidebar.slider(
    "Matching standards", value=2, min_value=1, max_value=5, key=6
)


# defining some global variables
men_population = int(men_percent * population * 0.01)
women_population = int(women_percent * population * 0.01)

# defining the main simulate function


@st.cache(persist="True")
def simulate(
    matches_per_day_men,
    matches_per_day_women,
    number_of_days,
    matching_criteria,
):

    men_ratings = [0] * men_population
    # making a list where we store the ratings of men, randomly generate from 1-10
    for i in range(len(men_ratings)):
        men_ratings[i] = generate_random_rating()

    women_ratings = [0] * women_population
    for i in range(len(women_ratings)):
        women_ratings[i] = generate_random_rating()

    men_matches_count = [0] * men_population
    women_matches_count = [0] * women_population
    men_matches_list = [[] for _ in range((men_population))]
    women_matches_list = [[] for _ in range((women_population))]

    for k in range(number_of_days):  # 14 is like doing this for 14 days
        for i in range(len(men_matches_count)):  # iterate over all men
            current_rating = men_ratings[i]  # get current 'rating'
            searching_women = random_match_index(
                matches_per_day_men, women_population
            )  # get a list containing the index of  women
            for j in range(len(searching_women)):

                # for each of these women, check if the rating is within range for the man in question, if yes call it a match for both (applied check to see they're not already matched)
                ##here we end up using minimum of (men,women) standards

                if (
                    current_rating - matching_criteria
                    <= women_ratings[searching_women[j]]
                    <= current_rating + matching_criteria
                    and searching_women[j] not in men_matches_list[i]
                ):
                    men_matches_count[i] = men_matches_count[i] + 1
                    women_matches_count[searching_women[j]] = (
                        women_matches_count[searching_women[j]] + 1
                    )
                    men_matches_list[i].append(searching_women[j])
                    women_matches_list[searching_women[j]].append(i)

        for i in range(len(women_matches_count)):
            current_rating = women_ratings[i]
            searching_men = random_match_index(matches_per_day_women, men_population)

            for j in range(len(searching_men)):
                if (
                    current_rating - matching_criteria
                    <= men_ratings[searching_men[j]]
                    <= current_rating + matching_criteria
                    and searching_men[j] not in women_matches_list[i]
                ):
                    women_matches_count[i] = women_matches_count[i] + 1
                    men_matches_count[searching_men[j]] = (
                        men_matches_count[searching_men[j]] + 1
                    )
                    women_matches_list[i].append(searching_men[j])
                    men_matches_list[searching_men[j]].append(i)
    return men_matches_count, women_matches_count


# helper functions - push to a helper.py for bettter readability
def generate_random_rating():
    return random.randrange(10) + 1


# helper functions - push to a helper.py for bettter readability
# this function generates a list of random indexes which act as potential matches
def random_match_index(matches_to_be_returned, searching_within):

    match_indexes = [0] * matches_to_be_returned
    for i in range(len(match_indexes)):
        match_indexes[i] = random.randrange(searching_within)
    return match_indexes


# calling the simulate function
men_matches_count, women_matches_count = simulate(
    matches_per_day_men,
    matches_per_day_women,
    number_of_days,
    matching_criteria,
)

# function to plot graphs


def women_plot():
    n, bins, patches = plt.hist(
        x=women_matches_count, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Women Data")

    maxfreq = n.max()
    st.pyplot()


def men_plot():
    n, bins, patches = plt.hist(
        x=men_matches_count, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Men Data")

    maxfreq = n.max()
    st.pyplot()


# ensure your arr is sorted from lowest to highest values first!
total_matches = men_matches_count + women_matches_count
total_matches = np.asarray(total_matches, dtype=np.float64)
np.sort(total_matches)


def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def lorenz(arr):
    # this divides the prefix sum by the total sum
    # this ensures all the values are between 0 and 1.0
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    # this prepends the 0 value (because 0% of all people have 0% of all wealth)
    return np.insert(scaled_prefix_sum, 0, 0)


lorenz_curve = lorenz(total_matches)


st.markdown(
    "I'd recently written an [article](https://harshj.substack.com/p/why-dating-apps-need-to-suck-to-make) about the business model of dating apps. A brief summary would be - dating apps monetize by offering services that allow people to get more matches. Men on average get very few matches on these apps and are 70 percent of the population on the app. I hypothesize that one big reason why men don't get matches on the app is the fact that the population ratios are skewed. I also suggest another business model based on a different incentive structure."
)

st.markdown(
    "Assume the dating app ecosystem was an economy, where the distributions of matches across people would be similar to distribution of wealth among people.  We can go on to see how 'unequal' the dating economy is using the Lorenz curve and the Gini coefficient."
)

st.markdown(
    "Getting to the point, I made a model of a dating app. In the next section I'll explain the model and talk about a couple of interesting things I noticed and the last section is about letting you play around with the model."
)

st.header("Model Details")

st.markdown("The core idea of the model can be summarised below :")
st.markdown(
    "1) Create a distribution of people who are randomly assinged 'attractiveness scores' between 1-10."
)
st.markdown("2) Show each person a set of randomly chosen people.")
st.markdown(
    "3) To match someone, we check if the other person's attractiveness score falls within a range of the person's attractiveness score. Say, person A has a rating of 7 and their matching standards are +-2 then anyone they see in the set mentioned above AND with a score between 5-9 will be counted as a match."
)

st.subheader("Parameter Details")
st.markdown(
    "1) Population : Define the number of people who'll be on the model of the dating app."
)
st.markdown(
    "2) Percentage of men : Define the percentage of men that will be on your dating app, the percentage of women would simply be 100- percentage of men. For model simplicity, I chose to assume a purely heterosexual society."
)
st.markdown(
    "3) Maximum number of matches (men) : This is the number of people any man will see on any particular day of the app (refer point 2 above), this theoretically also acts as the maximum number of matches a man can have in a day"
)
st.markdown(
    "4) Maximum number of matches (female) : The female verison of the above point"
)
st.markdown(
    "5) Number of days to run the simulation for : Self explanatory, but a day in this context refers to a timestamp. With every timestamp, people are shown a set of people and they may match with them."
)
st.markdown(
    "6) Matching standards : This is the range of attractiveness where someone may find a potential match. The standards are +/-, i.e X points above and X points below."
)

st.subheader("Interesting Results")

st.markdown("When I ran the model with the following parameters  :")
st.markdown("1) Population : 100000")
st.markdown("2) Percentage of men : 75")
st.markdown("3) Maximum number of matches (men) : 20")
st.markdown("4) Maximum number of matches (women) : 20")
st.markdown("5) Number of days to run the simulation for : 10")
st.markdown("6) Matching standards : 2")

st.markdown("I got a Gini coefficent of 0.304")

st.markdown(
    "To help put this into perspective, look at the below table. For example, if your gini coefficient is 0.50 - you're in the top 30 of the most UNEQUAL countries in the world. Remember, lower the gini score the more equal is your economy!"
)
data = {
    "Number of Countries in Range": [
        "0-30",
        "30-60",
        "60-90",
        "90-120",
        "120-150",
        "150-180",
    ],
    "Range (Gini-Coefficient)": [
        "0.63-0.45",
        "0.45-0.405",
        "0.405-0.357",
        "0.357-0.328",
        "0.328-0.274",
        "Below 0.274",
    ],
}


# Creates pandas DataFrame.
df = pd.DataFrame(data)
st.dataframe(df, width=1000)
st.markdown(
    "This is a slighly inequal economy, I'd have been a lot happier if the model came up with a value around 0.50 as that would be more inline with the current inquality. This difference could be due to a variety factors including but not limited to the choice of parameters and the core logic of the model. Or simply the reported inequality isn't an accurate value."
)

image = Image.open("men-75.png")
st.image(image, caption="Distribution of matches for men")
image = Image.open("women-75.png")
st.image(image, caption="Distribution of matches for women")


st.markdown(
    "Rerunning the same model with just one change, making the male-female population to 50:50 has some interesting results. The Gini coefficient drops to 0.10 which is a very equal economy. Infact the match distribution of men and women is almost the same too. This could be an indication of the hypothesis that all things the same, an improvement of sex ratio would lead to better outcomes for everyone on this app."
)
image = Image.open("men-50.png")
st.image(image, caption="Distribution of matches for men")
image = Image.open("women-50.png")
st.image(image, caption="Distribution of matches for women")


st.header("Try it yourself!")

st.markdown(
    "Play around with the parameters on the left side panel to see how inequality changes with various parameters. Simply change any of the parameters and the model will calculate and display the updated statistics!"
)

st.write("Simulated Gini Coefficient : ", gini(total_matches))


st.markdown("Below are the plots of the distribution of matches for men and women")
st.set_option(
    "deprecation.showPyplotGlobalUse", False
)  # just copied this to prevent a warning from occuring
men_plot()
women_plot()

st.header("Future Work and Conclusion")

st.markdown(
    "The model is far from 'finished', the current matching algorithm can definitely be tweaked and I can look into making the model more LGBTQ friendly."
)
st.markdown(
    "The most common saying in modeling is that all models are wrong, however some are useful. A useful insight from this modeling exercise would be that improved sex ratios are key to making better apps from a user perspective. The most important takeaway would be that modeling is an incredibly powerful skill,  knowing that we can partially replicate the complexities of human behaviour through a few lines of code is enough for me to learn more this field."
)
st.markdown(
    "If you want to send me some feedback about this model, or talk about mathematical models in general feel free to contact me [here](https://www.linkedin.com/in/harshjain0016/) or [here](https://twitter.com/harshh_jainn)."
)
