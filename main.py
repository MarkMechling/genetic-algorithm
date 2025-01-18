"""Step 1: Import Libraries"""
import tsplib95
import numpy as np
import random
import matplotlib.pyplot as plt

"""Step 2: Loading File"""
# Load TSP file and extract coordinates
def load_tsp(file_path):
    problem = tsplib95.load(file_path)
    coordinates = np.array([problem.node_coords[node] for node in sorted(problem.node_coords)])
    return coordinates

"""Step 3: Computing Distance Matrix and Calculating Distance"""
# Calculate distance matrix
def calculate_distance_matrix(coordinates):
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    return distance_matrix

# Calculate route distance
def calculate_distance(route, distance_matrix):
    return sum(
        distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)
    ) + distance_matrix[route[-1], route[0]]

"""Step 4: Genetic Algorithm"""
# Genetic algorithm for TSP
def genetic_algorithm(distance_matrix, population_size=50, generations=100, mutation_rate=0.1):
    num_cities = distance_matrix.shape[0]

    # Initialize population
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

    for _ in range(generations):
        # Evaluate fitness
        fitness_scores = [1 / calculate_distance(route, distance_matrix) for route in population]

        # Select parents (elitist selection)
        def elitist_selection(population, fitness_scores, num_elites=10):
            # Sort individuals by fitness in descending order
            sorted_indices = np.argsort(fitness_scores)[::-1]
            # Select the top individuals based on the number of elites
            elites = [population[i] for i in sorted_indices[:num_elites]]
            return elites

        parents = elitist_selection(population, fitness_scores)

        # Create next generation
        next_population = []
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            cut = random.randint(1, num_cities - 2)
            child = parent1[:cut] + [gene for gene in parent2 if gene not in parent1[:cut]]
            next_population.append(child)

        # Mutate
        for i in range(len(next_population)):
            if random.random() < mutation_rate:
                a, b = random.sample(range(num_cities), 2)
                next_population[i][a], next_population[i][b] = next_population[i][b], next_population[i][a]

        population = next_population

    # Return the best solution
    best_route = min(population, key=lambda route: calculate_distance(route, distance_matrix))
    best_distance = calculate_distance(best_route, distance_matrix)
    return best_route, best_distance

"""Step 5: Visualizing the Route"""
# Plot the route
def plot_route(coordinates, route, distance):
    route_coordinates = coordinates[route + [route[0]]]
    plt.figure(figsize=(10, 6))
    plt.plot(route_coordinates[:, 0], route_coordinates[:, 1], '-o', label=f"Distance: {distance:.2f}")
    plt.title("Best Route")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()

"""Step 6: Running the Main Function"""
# Main function
def main():
    dataset = str(input("Please enter the filename: "))
    file_path = fr"/Users/mark.mechling/PycharmProjects/Metaheuristics/tsp/{dataset}.tsp"

    # Load dataset
    coordinates = load_tsp(file_path)
    print(f"Loaded coordinates for {len(coordinates)} cities.")

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(coordinates)

    # Solve TSP with genetic algorithm
    print("Running Genetic Algorithm...")
    best_route, best_distance = genetic_algorithm(distance_matrix, population_size=50, generations=100)
    print(f"Best Distance: {best_distance:.2f}")
    print(f"Best Route: {best_route}")

    # Plot the best route
    plot_route(coordinates, best_route, best_distance)

if __name__ == "__main__":
    main()