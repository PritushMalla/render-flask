from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import re
app = Flask(__name__)


def convert_time_to_minutes(time_str):
    hours = 0
    minutes = 0

    # Match hours and minutes in the format PTXXHYYM
    time_match = re.match(r'PT(\d+)H(\d+)M', time_str)
    if time_match:
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
    else:
        time_match = re.match(r'PT(\d+)H', time_str)
        if time_match:
            hours = int(time_match.group(1))
        else:
            time_match = re.match(r'PT(\d+)M', time_str)
            if time_match:
                minutes = int(time_match.group(1))

    total_minutes = (hours * 60) + minutes
    return total_minutes
default_max_list = {
    'Calories': 2000,
    'FatContent': 100,
    'SaturatedFatContent': 13,
    'CholesterolContent': 300,
    'SodiumContent': 2300,
    'CarbohydrateContent': 325,
    'FiberContent': 40,
    'SugarContent': 40,
    'ProteinContent': 200
}

# Load the large dataset (replace with your actual file path)
data_chunks = []
chunksize = 10000  # Process 10,000 rows at a time
for chunk in pd.read_csv("/Users/Pritush/PycharmProjects/flask/.venv/lib/recipes.csv", chunksize=chunksize):
    data_chunks.append(chunk)
data = pd.concat(data_chunks, ignore_index=True)
data = data.dropna(subset=['CookTime', 'PrepTime', 'TotalTime'])
# Function to process input and recommend recipes
def recommend_recipes(user_input):

    dataset = data.copy()
    columns = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'RecipeIngredientParts',
               'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
               'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeInstructions']
    dataset = dataset[columns]

    # data['CookTime_minutes'] = data['CookTime'].apply(convert_time_to_minutes)
    # data['PrepTime_minutes'] = data['PrepTime'].apply(convert_time_to_minutes)
    # data['TotalTime_minutes'] = data['TotalTime'].apply(convert_time_to_minutes)
    relevant_columns = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']

    # Get the max nutritional values from user input
    max_nutritional_values = user_input

    # print(max_nutritional_values.items())
    if isinstance(max_nutritional_values, dict):
        print(f"User input max values: {max_nutritional_values}")

    # Filter dataset based on nutritional constraints
    filtered_data = data.copy()
    # print(filtered_data[relevant_columns].sample(n=10))

    filtered_data = dataset.copy()
    for i, column in enumerate(filtered_data.columns[6:15]):
        filtered_data = filtered_data[filtered_data[column] <= list(default_max_list.values())[i]]
    # print(filtered_data.columns
    #       )
    return filtered_data

            # Preprocess the data for nearest neighbor model
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(
        filtered_data.iloc[relevant_columns].to_numpy())
    # print("prepdata",prep_data) # assuming relevant columns start from index 6

    # Train the Nearest Neighbors model
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    # print("prepdata", prep_data)
    # Build the pipeline
    params = {'n_neighbors': 10, 'return_distance': False}
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])


    extracted_data = data.copy()
    for column, maximum in zip(extracted_data.columns[6:15], max_nutritional_values):
        extracted_data = extracted_data[extracted_data[column] < maximum]
    # Get the input data to compare (just an example of comparison)
    recommended_indices = pipeline.transform(max_nutritional_values)[0]
    recommendations = extracted_data.iloc[recommended_indices]
    # Get recommendations from filtered data
    pd.set_option('display.max_rows', None)  # Or a large number like 10000

    # Show all columns
    pd.set_option('display.max_columns', None)  # Or a large number

    # Show full column width (important for RecipeInstructions)
    pd.set_option('display.max_colwidth', None)  # Or a large number like 500

    # Show all values in a column (no truncation)
    pd.set_option('display.expand_frame_repr', True)
    print("recommendation",recommendations)
    print(recommendations[['Name', 'Calories', 'FatContent', 'ProteinContent']])


    return recommendations[['Name', 'Calories', 'FatContent', 'ProteinContent']]


@app.route('/recommend', methods=['GET','POST'])

def recommend():
  # Get data from Flutter app
    user_input = request.get_json()
    #print(user_input)
  # Get recommendations
    recommendations = recommend_recipes(user_input)

    if isinstance(recommendations, pd.DataFrame):
        # Select the top 10 rows and specific columns
        top_10_subset = recommendations.sample(n=10, random_state=None).iloc[:, [0, 1, 15]]

        print("Top 10 subset:\n", top_10_subset)

        # Jsonify the top 10 rows
        print("Jsonified data:", jsonify(top_10_subset.to_dict(orient='records')))

        # Return top 10 recommendations as JSON
        return jsonify(top_10_subset.to_dict(orient='records'))

        # Handling case when recommendations is a list
    elif isinstance(recommendations, list):
        recommendations = recommendations[:10]
        print("Recommendations:", recommendations)

        # Send recommendations as JSON
        return jsonify(recommendations)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
