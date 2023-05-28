from keras.models import Sequential, clone_model, Model
from keras.layers import Dense, Dropout, Conv1D, LSTM, Input, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import binary_crossentropy, categorical_crossentropy
import numpy as np
import random
from multiprocessing import Pool
from functools import partial

class BiSOLA:
    def __init__(self, input_dim, num_tasks, population_size):
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.population_size = population_size
        self.population = [self.create_initial_model() for _ in range(population_size)]
        self.activation_functions = ['relu', 'tanh', 'sigmoid', 'softmax', 'elu', 'selu', 'softplus', 'softsign', 'linear']
        self.loss_functions = [binary_crossentropy, categorical_crossentropy]
        self.optimizers = [Adam, RMSprop, SGD]

    def create_initial_model(self):
        # Image submodel
        image_input = Input(shape=(28, 28, 1))  # Example input shape for image data (e.g. MNIST)
        x = Conv1D(32, kernel_size=3, activation='relu')(image_input)
        x = Flatten()(x)
        image_output = Dense(64, activation='relu')(x)
        image_submodel = Model(inputs=image_input, outputs=image_output)

        # Time series submodel
        ts_input = Input(shape=(100, 1))  # Example input shape for time series data: 100 time steps
        x = LSTM(32, activation='relu')(ts_input)
        ts_output = Dense(64, activation='relu')(x)
        ts_submodel = Model(inputs=ts_input, outputs=ts_output)

        # Shared layers
        shared_input = Input(shape=(64,))
        x = Dense(32, activation='relu')(shared_input)
        shared_output = Dense(self.num_tasks, activation='sigmoid')(x)

        # Final model
        image_final = Model(inputs=image_input, outputs=shared_output(image_submodel.model.output))
        ts_final = Model(inputs=ts_input, outputs=shared_output(ts_submodel.model.output))

        return {"image_model": image_final, "ts_model": ts_final}

    def adaptive_architecture(self, models, X, y):
        for key, model in models.items():
            accuracy = np.mean([model.evaluate(X[key][i], y[i], verbose=0)[1] for i in range(self.num_tasks)])
            if accuracy < 0.8 and len(model.layers) < 10:
                new_layer = self.create_new_layer()
                if isinstance(new_layer, Dropout):
                    # Add a Dense or Conv1D layer before Dropout
                    preceding_layer = random.choice(['Dense', 'Conv1D'])
                    if preceding_layer == 'Dense':
                        model.add(Dense(random.choice([16, 32, 64]), activation=random.choice(self.activation_functions)))
                    elif preceding_layer == 'Conv1D':
                        model.add(Conv1D(random.choice([16, 32, 64]), kernel_size=random.choice([3, 5, 7]), activation=random.choice(self.activation_functions)))
                model.add(new_layer)
            models[key] = model
        return models

    def create_new_layer(self):
        layer_type = random.choice(['Dense', 'Dropout', 'Conv1D', 'LSTM'])
        if layer_type == 'Dense':
            return Dense(random.choice([16, 32, 64]), activation=random.choice(self.activation_functions))
        elif layer_type == 'Dropout':
            return Dropout(random.uniform(0.1, 0.5))
        elif layer_type == 'Conv1D':
            return Conv1D(random.choice([16, 32, 64]), kernel_size=random.choice([3, 5, 7]), activation=random.choice(self.activation_functions))
        elif layer_type == 'LSTM':
            return LSTM(random.choice([16, 32, 64]), activation=random.choice(self.activation_functions))

    def self_organize(self, models, X, y):
        for key, model in models.items():
            new_models = []

            # Layer removal
            for i in range(1, len(model.layers) - 1):
                new_model = clone_model(model)
                new_model.set_weights(model.get_weights())
                new_model.pop(i)
                new_model.compile(loss=model.loss, optimizer=model.optimizer, metrics=['accuracy'])
                new_models.append(new_model)

            # Layer type changes
            for i in range(1, len(model.layers) - 1):
                new_model = clone_model(model)
                new_model.set_weights(model.get_weights())
                new_layer = self.create_new_layer()
                new_model.layers[i] = new_layer
                new_model.compile(loss=model.loss, optimizer=model.optimizer, metrics=['accuracy'])
                new_models.append(new_model)

            # Layer reordering
            reordered_indices = list(range(1, len(model.layers) - 1))
            for _ in range(10):
                random.shuffle(reordered_indices)
                new_model = clone_model(model)
                new_model.set_weights(model.get_weights())
                new_layers = [new_model.layers[i] for i in reordered_indices]
                new_model.layers[1:-1] = new_layers
                new_model.compile(loss=model.loss, optimizer=model.optimizer, metrics=['accuracy'])
                new_models.append(new_model)

            # Evaluate all new models and the original model
            accuracies = [np.mean([m.evaluate(X[key][i], y[i], verbose=0)[1] for i in range(self.num_tasks)]) for m in new_models + [model]]
            best_model_index = np.argmax(accuracies)

            if best_model_index == len(new_models):
                models[key] = model
            else:
                models[key] = new_models[best_model_index]
        return models

    def crossover(self, parent_models1, parent_models2):
        child_models = {}
        for key in parent_models1.keys():
            parent1 = parent_models1[key]
            parent2 = parent_models2[key]
            child = clone_model(parent1)
            child.set_weights(parent1.get_weights())

            for i in range(len(child.layers)):
                if random.random() < 0.5:
                    child.layers[i].set_weights(parent2.layers[i].get_weights())

            child.compile(loss=random.choice(self.loss_functions), optimizer=random.choice(self.optimizers)(), metrics=['accuracy'])
            child_models[key] = child
        return child_models

    def generate_new_population(self, X, y):
        eval_data = {}
        for key in self.population[0].keys():
            eval_data[key] = {}
            eval_data[key]["X"] = [X[key][i] for i in range(self.num_tasks)]
            eval_data[key]["y"] = y

        accuracies = [
            np.mean([
                np.mean([model[key].evaluate(eval_data[key]["X"][i], eval_data[key]["y"][i], verbose=0)[1] for i in range(self.num_tasks)])
                for key in model.keys()
            ]) for model in self.population
        ]

        sorted_indices = np.argsort(accuracies)[-self.population_size // 2:]
        new_population = [self.population[i] for i in sorted_indices]
        for i in range(self.population_size // 2):
            parent1 = random.choice(new_population)
            parent2 = random.choice(new_population)
            new_population.append(self.crossover(parent1, parent2))
        self.population = new_population

    def mutate(self, models, X, y):
        for key, model in models.items():
            new_model = clone_model(model)
            new_model.set_weights(model.get_weights())
            mutation_choice = random.choice(['activation', 'optimizer', 'loss', 'architecture'])

            if mutation_choice == 'activation':
                layer_num = random.randint(0, len(new_model.layers) - 1)
                new_activation = random.choice(self.activation_functions)
                new_model.layers[layer_num].activation = new_activation
            elif mutation_choice == 'optimizer':
                new_optimizer = random.choice(self.optimizers)
                new_model.compile(loss=model.loss, optimizer=new_optimizer(), metrics=['accuracy'])
            elif mutation_choice == 'loss':
                new_loss = random.choice(self.loss_functions)
                new_model.compile(loss=new_loss, optimizer=model.optimizer, metrics=['accuracy'])
            elif mutation_choice == 'architecture':
                new_layer = self.create_new_layer()
                random_position = random.choice(range(len(new_model.layers)))
                new_model.layers.insert(random_position, new_layer)

            new_model.compile(loss=new_model.loss, optimizer=new_model.optimizer, metrics=['accuracy'])

            accuracy_old = np.mean([model.evaluate(X[key][i], y[i], verbose=0)[1] for i in range(self.num_tasks)])
            accuracy_new = np.mean([new_model.evaluate(X[key][i], y[i], verbose=0)[1] for i in range(self.num_tasks)])

            if accuracy_new > accuracy_old:
                models[key] = new_model
            else:
                models[key] = model

        return models

    def fit(self, X, y, epochs=10, parallel=False):
        if parallel:
            self.parallel_fit(X, y, epochs=epochs)
            return

        for models in self.population:
            for key, model in models.items():
                for i in range(self.num_tasks):
                    model.fit(X[key][i], y[i], epochs=epochs, verbose=0)
            models = self.adaptive_architecture(models, X, y)
            models = self.self_organize(models, X, y)
            models = self.mutate(models, X, y)
        self.generate_new_population(X, y)

    def fit_model(self, models, X, y, epochs):
        for key, model in models.items():
            for i in range(self.num_tasks):
                model.fit(X[key][i], y[i], epochs=epochs, verbose=0)
        models = self.adaptive_architecture(models, X, y)
        models = self.self_organize(models, X, y)
        models = self.mutate(models, X, y)
        return models

    def parallel_fit(self, X, y, epochs=10):
        # Using a Pool of workers, with one worker per model in the population.
        with Pool(processes=len(self.population)) as pool:
            # The map function applies the fit_model function to each model in the population.
            # Using partial to create a new function that has the X, y, and epochs parameters pre-filled.
            self.population = pool.map(partial(self.fit_model, X=X, y=y, epochs=epochs), self.population)

        self.generate_new_population(X, y)
