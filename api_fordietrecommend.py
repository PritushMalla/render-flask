from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__)

# Load the large dataset (replace with your actual file path)
data = pd.read_csv("/Users/Pritush/PycharmProjects/flask/.venv/lib/recipes.csv")


# Function to process input and recommend recipes
def recommend_recipes(user_input):
    # Get the max nutritional values from user input
    max_nutritional_values = user_input['max_nutritional_values']

    # Filter dataset based on nutritional constraints
    filtered_data = data.copy()
    for nutrient, value in max_nutritional_values.items():
        filtered_data = filtered_data[filtered_data[nutrient] <= value]

    # Preprocess the data for nearest neighbor model
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(
        filtered_data.iloc[:, 6:15].to_numpy())  # assuming relevant columns start from index 6

    # Train the Nearest Neighbors model
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)

    # Create a pipeline for preprocessing and recommendation
    transformer = FunctionTransformer(neigh.kneighbors, kw_args={'n_neighbors': 10})
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])

    # Get the input data to compare (just an example of comparison)
    recipe_input = [user_input['max_nutritional_values'].values()]
    recommended_indices = pipeline.transform(recipe_input)[0]

    # Get recommendations from filtered data
    recommendations = filtered_data.iloc[recommended_indices]

    # Return recommendations
    return recommendations[['Name', 'Calories', 'FatContent', 'ProteinContent']]


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get data from Flutter app
    user_input = request.json

    # Get recommendations
    recommendations = recommend_recipes(user_input)

    # Send recommendations as JSON
    return jsonify(recommendations.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
