from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_demand():
    num_samples = request.json.get('num_samples', 10)
    random_input = np.random.normal(0, 1, (num_samples, noise_dim))
    generated_demand = generator.predict(random_input)
    generated_demand = scaler.inverse_transform(generated_demand)
    return jsonify(generated_demand.tolist())

if __name__ == '__main__':
    app.run(debug=True)