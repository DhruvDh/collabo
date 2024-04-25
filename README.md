# collabo

This project aims to optimize the collaborative structure of LLMTeams using a genetic algorithm to maximize task completion efficiency and economic viability.

The genetic algorithm evolves a population of LLMTeamGenome instances, each representing a different configuration of genes (LLMGene). The fitness of each LLMTeamGenome is evaluated based on the performance of the LLMTeam it represents on the given set of tasks.

The performance metrics include task completion rate, collaboration efficiency, and economic viability (considering the costs and rewards associated with solving the tasks). Through the process of selection, crossover, and mutation, the genetic algorithm explores different combinations of genes and converges towards the optimal values that yield the most effective collaborative structures for the LLMTeams.

## LLM

The LLM struct represents an individual language model with its specific characteristics and capabilities.

```rust
struct LLM {
    context_length: usize,
    reasoning_ability: f32,
    planning_ability: f32,
    verbosity: f32,
    competency_1: f32,
    competency_2: f32,
    competency_3: f32,
    cost_per_input_token: f32,
    cost_per_output_token: f32,
}
```

- `context_length`: The maximum number of tokens the LLM can process in a single context.
- `reasoning_ability`: A float value indicating the LLM's ability to reason and draw logical conclusions.
- `planning_ability`: A float value representing the LLM's ability to plan and break down tasks into subtasks.
- `verbosity`: A float value indicating the number of tokens required to complete a task.
- `competency_1`: A float value indicating the overall competency of the LLM in solving tasks.
- `competency_2`: A float value indicating the overall competency of the LLM in solving tasks.
- `competency_3`: A float value indicating the overall competency of the LLM in solving tasks.
- `cost_per_input_token`: The cost associated with processing each input token.
- `cost_per_output_token`: The cost associated with generating each output token.

## LLMTeam

The LLMTeam enum represents a team of collaborating LLMs, which can be a single LLM, a vertical collaboration, or a horizontal collaboration.

```rust
enum LLMTeam {
    Single(LLM),
    Vertical {
        leader: Box<LLMTeam>,
        followers: Vec<LLMTeam>,
    },
    Horizontal {
        members: Vec<LLMTeam>,
    },
    Invalid,
}
```

- `Single`: Represents an individual LLM working independently.
- `Vertical`: Represents a vertical collaboration strategy, where a leader LLMTeam directs and coordinates the work of follower LLMTeams.
- `Horizontal`: Represents a horizontal collaboration strategy, where multiple LLMTeams work together as equals to solve tasks.
- `Invalid`: Represents an invalid LLM, for error handling purposes. This variant will always get a fitness score of 0.

## LLMGenome

The LLMTeamGenome struct represents the genome of an LLMTeam in the NEAT framework. It includes the genotype string representation.

```rust
struct LLMTeamGenome {
    genotype: String,
}
```

### Genotype

The genotype is a string representation of the collaborative structure of an LLMTeam. It follows these rules:

- All 'S' characters represent single LLM team members.
- The first 'S' character dictates that there is only one single LLM working independently.
- Any number of 'S' characters that come after a 'V' character work under that 'V' LLM in a vertical hierarchy.
- Any number of 'S' characters that come after an 'H' character work with that 'H' LLM in a horizontal fashion, where they all work in parallel and then vote for a solution.

The genotype string is parsed using the following algorithm:

1. If the genotype string is empty, return `LLMTeam::Invalid`.
2. If the first character of the genotype is 'S':
   - If the genotype length is 1, return `LLMTeam::Single(...)`.
   - Otherwise, return `LLMTeam::Invalid` (as 'S' should only appear alone or after 'V' or 'H').
3. Initialize an empty stack called `team_stack` to store the parsed team structures.
4. Traverse the genotype string from right to left:
   - If the current character is 'S', push `LLMTeam::Single(...)` onto the `team_stack`.
   - If the current character is 'H':
     - Create an empty vector called `members`.
     - While `team_stack` is not empty and the top of `team_stack` is not `LLMTeam::Vertical`, pop elements from `team_stack` and push them onto `members`.
     - Push `LLMTeam::Horizontal { members }` onto the `team_stack`.
   - If the current character is 'V':
     - Create an empty vector called `followers`.
     - While `team_stack` is not empty and the top of `team_stack` is not `LLMTeam::Horizontal`, pop elements from `team_stack` and push them onto `followers`.
     - If `team_stack` is empty, return `LLMTeam::Invalid` (as there should be a leader for the vertical team).
     - Pop the top element from `team_stack` and assign it to `leader`.
     - Push `LLMTeam::Vertical { leader, followers }` onto the `team_stack`.
   - If the current character is none of the above, return `LLMTeam::Invalid`.
5. After the traversal, if the `team_stack` contains exactly one element, return that element as the final team structure.
6. If the `team_stack` contains more than one element or is empty, return `LLMTeam::Invalid`.

## Tasks

The Tasks struct represents a set of tasks that the LLMTeams need to solve. Each task has specific requirements and characteristics.

```rust
struct Task {
    id: String,
    description: String,
    reasoning_required: f32,
    planning_required: f32,
    competency_required: f32,
    token_estimate: usize,
    economic_value: f32,
}
```

- `id`: A unique identifier for the task.
- `description`: A detailed description of the task.
- `reasoning_required`: A float value indicating the level of reasoning required to solve the task.
- `planning_required`: A float value representing the level of planning required to solve the task.
- `competency_required`: A float value indicating the overall competency required to solve the task.
- `token_estimate`: An estimate of the number of tokens required to complete the task.
- `economic_value`: The economic value generated upon successful completion of the task.

## Scratch

- "H(S2)(S1)(S3)" could represent a horizontal team with three LLMs, where their votes are weighted as 2, 1, and 3, respectively.
- Prefix genes that change how the input list of LLMs is sorted.

## Simulation

### Initialization

### Evaluation

### Selection

### Crossover

### Mutation
