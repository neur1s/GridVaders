# Space Invaders (aka GridVaders)
<div align="center">
  <img width="200" height="162" alt="GridVaders banner" src="https://github.com/user-attachments/assets/770fd05d-eb85-4771-a135-8676553c1822" />
</div>

## Comparing neural networks performing spatial navigation in a 2D gridlike world

This project explores and compares the performance of different recurrent neural network (RNN) architectures—Vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)—on a spatial navigation task. The task is an n-back task set in a 2D grid world, where the model has to predict its location 'n' steps ago. This repository contains the code for implementing the models, running the experiments, and analyzing the results.

### Part of Neuromatch's NeuroAI 2025 Academy

## Features

*   Implementation of Vanilla RNN, LSTM, and GRU models for the n-back spatial task.
*   Jupyter notebooks for demonstrating and exploring the models' behavior.
*   Scripts for generating training data and running experiments.
*   Analysis and visualization of model performance and hidden state representations using t-SNE.

## Repository Structure

-   `n_back_spatial_task.py`: Core script to generate the n-back spatial task dataset.
-   `lstm.ipynb`, `VanillaRNN.ipynb`, `gru_explorer.py`: Main notebooks and scripts for training and evaluating the LSTM, Vanilla RNN, and GRU models.
-   `LSTM functions/`, `VanillaRNN functions/`: Directories containing helper functions, demo notebooks, and plotting scripts for each model.
-   `analysis_plots.py`: Script for generating plots for analysis.
-   `results/`: Directory to store the results of the experiments.
-   `requirements.txt`: A list of all the python packages required to run the code.

## Getting Started


### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/GridVaders.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd GridVaders
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary way to interact with this project is through the Jupyter notebooks. You can start Jupyter Lab by running:

```bash
jupyter lab
```

Then, you can open and run the notebooks (`lstm.ipynb`, `VanillaRNN.ipynb`, etc.) to see the models in action.

## Results

The notebooks contain the code to train the models and visualize the results. The `results/` directory is intended to store the outputs of the experiments, such as trained model weights and performance metrics. The analysis includes accuracy plots across different 'n' values in the n-back task and t-SNE visualizations of the RNN hidden states to understand how the models represent the grid space.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors:
* Ítalo Karmann Aventurato, PhD Candidate (Universidade Estadual de Campinas, BR)
* Sofiya Zbaranska, PhD Candidate (University of Toronto, CA)
* Arthur Rodrigues, MSc Student (Universidade Federal de Minas Gerais, BR)
* Setareh Metanat, BSc in Cognitive Science, Research Assistant (University of California San Diego, USA)
* [Didi Ramsaran Chin](https://neurodidi.github.io/), BSc in Psychology (Universidad Católica Andrés Bello, VE)
* Sara Asadi, MSc Student (University of Lethbridge, CA)
* [Muhammad Mushhood Ur Rehman](https://www.linkedin.com/in/RehmanMushhood), Masters in Public Health Student (University of Edinburgh, UK)
* Kate Tabor, PhD in Neuroscience (University of Washington, USA)
* Souvik Bhattacharya, Ph.D. Student in Computer Science (University of Illinois, USA)
* Leticia Cid, BASc, Research Assistant (University of British Columbia, CA)

### Project TA
Mostafa Miandary Hossein, PhD Student (University of Toronto, CA)

### Regular TA
Karthika Kamath, PhD Student (University of Birmingham, UK)
