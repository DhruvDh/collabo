# collabo

This project aims to optimize the collaborative architecture of multiple LLM teams working together using a genetic algorithm to maximize task completion efficiency and economic viability. The genetic algorithm evolves a population of `LlmTeamGenome` instances, each representing a different configuration of genes (`LLMGene`).

The fitness of each `LlmTeamGenome` is evaluated based on the performance of the `LlmTeam` it represents on the given set of tasks. Through the process of selection, crossover, and mutation, the genetic algorithm explores different combinations of genes and converges towards the optimal values that yield the most effective collaborative structures for the LLMTeams.

## Motivation

- Large Language Models (LLMs) are generally effective in solving a wide range of tasks.
- The more capable LLMs are very expensive to train and run.
- It would be ideal if many smaller LLMs could collaborate to solve tasks more efficiently, perhaps outperforming a single large LLM.
- How to architect the collaboration of multiple LLMs to maximize task completion efficiency is unclear.
- Our work covers three types of collaboration structures: single, vertical, and horizontal.
  - Single: Each LLM works independently.
  - Vertical: A leader LLM coordinates the work of follower LLMs. The leader breaks down tasks and assigns subtasks to followers, which may or may not be easier to solve.
  - Horizontal: Multiple LLMs work together as equals. All work in parallel, if the majority "succeeds", the task is considered successful.

## Genes and Genotype

- What should be the genes?
- Our research question is related to "how" many LLMs should collaborate, so the genes should represent the collaboration structure.
- We come up with a string representation of a tree-like collaboration structure -

  ```
  SHSSVSS
  ```

- If the first gene is an `S`, it means the entire team is a single LLM, other genes are ignored.
- If the first gene is not an `S`:
  - We reverse the string and start from the right.
  - As we come across `S`s, we add them to a stack.
  - As soon as we come across a `V` or an `H`, we add all the prior `S`s to the `V` or `H` as followers.
- So our genes describe a `LlmFactory` - a way to create a `LlmTeam`.

## Fitness Function

- Our genes represent a `LlmFactory`, so it must be evaluated by giving it `LLMs` to arrange into a team.
- In our calculate fitness function:
  - For every task, we randomly initialize `N` `Llm`s.
  - These LLMs are arranged into teams according to the Genotype, N times.
  - Each team tries to solve the task,
    - If successful, the fitness function is incremented by one.
    - Otherwise, no action is taken.

## Solving Tasks

- Each `Llm` has a set of characteristics that determine its performance.

    ```rs
    struct Llm {
        context_length: usize,
        planning_ability: f32,
        verbosity: f32,
        competency: f32,
        cost_per_token: f32,
    }
    ```

- Each task has a set of requirements that must be met for it to be solved.

    ```rs
    struct Task {
        root_task: SubTask,
        subtasks: Vec<SubTask>,
        economic_value: f32,
        llms: Vec<Vec<Llm>>,
    }

    struct SubTask {
        reasoning_required: f32,
        planning_required: f32,
        competency_required: f32,
        token_estimate: usize,
    }
    ```

- There is also an `LLMTeam` struct, which can be a single LLM, a vertical collaboration, or a horizontal collaboration.

    ```rs
    enum LlmTeam {
        Single(Llm),
        Vertical {
            leader: Box<LlmTeam>,
            followers: Vec<LlmTeam>,
        },
        Horizontal {
            members: Vec<LlmTeam>,
        },
        Invalid,
    }
    ```

- If the LlmTeam is invalid, it always gets a fitness score of 0. (fails task)
- If the LlmTeam is a single LLM, the task is attempted by that LLM. (more on this in the next section)
- If the LlmTeam is a vertical collaboration, the leader LLM breaks down the task and assigns subtasks to followers.
  - If 75% of the followers succeed, the task is considered successful.
- If the LlmTeam is a horizontal collaboration, all LLMs work in parallel.
  - If >50% of the LLMs succeed, the task is considered successful.
  - I realize that this is a mistake as I write this.

## Solving Tasks with Single LLMs

```rs
    fn solve_task(&self, task: &Task) -> bool {
        let scaled_competency = self.competency * task.root_task.competency_required;
        let scaled_planning_ability = self.planning_ability * task.root_task.planning_required;
        let avg_competency = (scaled_competency + scaled_planning_ability) / 2.0;

        let scaled_token_estimate =
            (task.root_task.token_estimate as f32 * self.verbosity) as usize;
        let adjusted_competency = if scaled_token_estimate > self.context_length {
            avg_competency / 2.0
        } else {
            avg_competency
        };

        let mut rng = SmallRng::from_entropy();

        Normal::new(adjusted_competency as f64, 0.2)
            .unwrap()
            .sample(&mut rng)
            > 0.5
    }
```

## Breaking down tasks into subtasks

```rs
    fn break_down_task(&self, task: &Task) -> Vec<SubTask> {
        let threshold =
            1.0 - (task.root_task.reasoning_required + task.root_task.planning_required) / 2.0;
        let mut rng = SmallRng::from_entropy();
        let success = Normal::new(self.planning_ability, 0.1)
            .unwrap()
            .sample(&mut rng)
            > threshold;

        if success {
            task.subtasks.clone()
        } else {
            task.subtasks
                .iter()
                .map(|subtask| SubTask {
                    reasoning_required: (subtask.reasoning_required + 0.1).min(1.0),
                    planning_required: (subtask.planning_required + 0.1).min(1.0),
                    competency_required: (subtask.competency_required + 0.1).min(1.0),
                    ..subtask.clone()
                })
                .collect()
        }
    }
```

## Simulation

The genetic algorithm simulation consists of the following steps:

1. **Initialization**: Generate an initial population of `LlmTeamGenome` instances with random gene configurations.

2. **Evaluation**: Evaluate the fitness of each `LlmTeamGenome` by assessing the performance of the corresponding `LlmTeam` on a set of tasks. The fitness is calculated based on the task completion rate, collaboration efficiency, and economic viability.

3. **Selection**: Select the fittest individuals from the population to serve as parents for the next generation. The selection method used is tournament selection.

4. **Crossover**: Create offspring by applying the cycle crossover method to the selected parents. The crossover probability is adaptively adjusted based on the progress of the optimization.

5. **Mutation**: Introduce random variations in the offspring by applying the swap mutation method. The mutation probability is adaptively adjusted based on the progress of the optimization.

6. **Survivor Selection**: Select the individuals that will survive to the next generation based on their fitness. The survivor selection method used is fitness-based.

7. **Termination**: Repeat steps 2-6 for a specified number of generations or until a satisfactory solution is found.

The genetic algorithm parameters, such as population size, number of generations, crossover and mutation probabilities, and selection methods, can be adjusted to fine-tune the optimization process.

## Usage

To run the LLMTeam optimization using the genetic algorithm, execute the `main` function in the provided Rust code. The program will simulate the optimization process and display the best individual (LLMTeam configuration) found at the end of the simulation.

You can customize the simulation parameters, such as the maximum number of generations, by passing command-line arguments. For example, to set the maximum number of generations to 20, run the program with the following command:

```shell
cargo run -- -m 20

[2024-04-25T15:53:36Z INFO  genetic_algorithms::ga] Generation number: 999
[2024-04-25T15:53:36Z INFO  genetic_algorithms::ga] Generation number: 1000
Best individual: [LlmTeamGenome { dna: [Single(0), Horizontal(5), Horizontal(2), Vertical(4), Single(3), Vertical(1)], age: 0, fitness: 327970.11999999976 }]
```

The program will output the progress of the optimization process and the best individual found at the end of the simulation.
