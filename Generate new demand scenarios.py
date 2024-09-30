num_samples = 10
random_input = np.random.normal(0, 1, (num_samples, noise_dim))
generated_demand = generator.predict(random_input)

# Rescale the generated sales back to the original range
generated_demand = scaler.inverse_transform(generated_demand)

print(generated_demand)