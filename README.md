# Space Invaders (aka GridVaders)

## Neuromatch NeuroAI 2025 Academy Project: Comparing neural networks performing spatial memory task in a 2D grid world

This project explores and compares the performance of different recurrent neural network (RNN) architectures—Vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)—on a synthetic spatial memory task. The models are trained on an N-back task set in a 2D grid world, where the model takes random moves (up, down, left or right) in a 5x5 grid world and has to recall its location 'N' steps ago. This repository provides the complete codebase for model implementation, experimental execution, and analysis of memory capacity, spatial representations, and cross-model differences in these representations.

Access our presentation slides [here](https://github.com/neurodidi/GridVaders/blob/ca9f7c682b280b5c31489a3e2cfb357889447a06/2025-07%20NeuroAI%20Space%20Invaders%20Presentation.pdf).

<div align="center">
  <img alt="GridVaders banner" src="https://github.com/user-attachments/assets/4913e710-1833-4ab3-b81e-6013b3ba903c" />
</div>

## Features

*   Implementation of Vanilla RNN, LSTM, and GRU models for the n-back spatial task.
*   Jupyter notebooks for demonstrating and exploring the models' behavior and hidden state representations.
*   Scripts for generating training data, training and evaluation the models.
*   Analysis and visualization of model performance and hidden state representations using t-SNE and representational similarity analysis.

## Results

The notebooks contain the code to train, test the models, and visualize the results. The analysis includes accuracy plots across different 'N' values in the N-back task, t-SNE/PCA visualizations of the RNN hidden states to understand how the models represent the grid space, and Pearson correlation analyses to assess whether the similarity between hidden representations reflects the physical distances between locations in the gridworld (whether spatial structure is preserved in the neural network embeddings). In addition, we performed representational similarity analysis to examine how these spatial representations compare across the RNNs. The example results below demonstrate that the models successfully learned both the memory task and the spatial structure of the grid world (fully implicitly!), with LSTMs and GRUs exhibiting highly similar representations.

<div align="center">
  <img alt="Results" src="https://github.com/user-attachments/assets/39d762f2-8a90-451d-93d0-b6ff96d94c17" />
</div>

## Repository Structure

-   `n_back_spatial_task.py`: Core script to generate the n-back spatial task dataset.
-   `LSTM_demo.ipynb`, `VanillaRNN_demo.ipynb`, `GRU_demo.ipynb`: Main notebooks demonstrating training and evaluating the LSTM, Vanilla RNN, and GRU models.
-   `LSTM functions/`, `VanillaRNN functions/`, `GRU functions/`: Directories containing helper functions, and plotting scripts for each model.
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

Then, you can open and run the notebooks (`LSTM_demo.ipynb`, `VanillaRNN_demo.ipynb`, etc.) to see the models in action. Ensure that each notebook resides in the same directory as its corresponding helper function scripts to allow for seamless code execution.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors:
* [Sofiya Zbaranska](http://github.com/neur1s), PhD Candidate (University of Toronto, CA)
* Ítalo Karmann Aventurato, PhD Candidate (Universidade Estadual de Campinas, BR)
* Arthur Rodrigues, MSc Student (Universidade Federal de Minas Gerais, BR)
* Setareh Metanat, BSc in Cognitive Science, Research Assistant (University of California San Diego, USA)
* [Didi Ramsaran Chin](https://neurodidi.github.io/), BSc in Psychology (Universidad Católica Andrés Bello, VE)
* [Sara Asadi](https://github.com/saraasadi78), MSc Student (University of Lethbridge, CA)
* [Muhammad Mushhood Ur Rehman](https://www.linkedin.com/in/RehmanMushhood), Masters in Public Health Student (University of Edinburgh, UK)
* Kate Tabor, PhD in Neuroscience (University of Washington, USA)
* Souvik Bhattacharya, Ph.D. Student in Computer Science (University of Illinois, USA)
* Leticia Cid, BASc, Research Assistant (University of British Columbia, CA)

### Project TA
Mostafa Miandary Hossein, PhD Student (University of Toronto, CA)

### Regular TA
Karthika Kamath, PhD Student (University of Birmingham, UK)
