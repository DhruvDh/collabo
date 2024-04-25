//! # LLMTeam Optimization using Genetic Algorithm
//!
//! This project aims to optimize the collaborative structure of LLMTeams using a genetic algorithm
//! to maximize task completion efficiency and economic viability. The genetic algorithm evolves a
//! population of LLMTeamGenome instances, each representing a different configuration of genes (LLMGene).
//!
//! The fitness of each LLMTeamGenome is evaluated based on the performance of the LLMTeam it represents
//! on the given set of tasks. The performance metrics include task completion rate, collaboration efficiency,
//! and economic viability (considering the costs and rewards associated with solving the tasks).
//!
//! Through the process of selection, crossover, and mutation, the genetic algorithm explores different
//! combinations of genes and converges towards the optimal values that yield the most effective collaborative
//! structures for the LLMTeams.

use genetic_algorithms::{
    configuration::ProblemSolving,
    ga,
    operations::{Crossover, Mutation, Selection, Survivor},
    population::Population,
    traits::ConfigurationT,
};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use statrs::distribution::{Bernoulli, Exp, Gamma};

use genetic_algorithms::traits::{GeneT, GenotypeT};

// Constants for task and subtask generation
const TASK_DIFFICULTY_MEAN: f64 = 0.8;
const TASK_DIFFICULTY_STD_DEV: f64 = 0.2;
const SUBTASK_TOKEN_ESTIMATE_MIN: usize = 1000;
const SUBTASK_TOKEN_ESTIMATE_MAX: usize = 200000;
const ROOT_TASK_TOKEN_ESTIMATE_MEAN: f64 = 24000.0;
const ROOT_TASK_TOKEN_ESTIMATE_STD_DEV: f64 = 15000.0;
const SUBTASK_COUNT_MIN: usize = 1;
const SUBTASK_COUNT_MAX: usize = 15;

/// Represents the different types of genes in the LLMTeamGenome.
#[derive(Clone, Copy, Default)]
enum LLMGene {
    #[default]
    /// Represents a single LLM instance in the team, working independently.
    Single,
    /// Represents a vertical collaboration structure with a leader and followers.
    Vertical,
    /// Represents a horizontal collaboration structure with multiple members.
    Horizontal,
}

impl GeneT for LLMGene {
    fn set_id(&mut self, id: i32) {
        *self = match id {
            0 => LLMGene::Single,
            1 => LLMGene::Vertical,
            2 => LLMGene::Horizontal,
            _ => panic!("Invalid gene ID"),
        };
    }

    fn get_id(&self) -> i32 {
        match self {
            LLMGene::Single => 0,
            LLMGene::Vertical => 1,
            LLMGene::Horizontal => 2,
        }
    }
}

/// Represents an individual language model with its specific characteristics and capabilities.
#[derive(Clone, Default, Copy)]
struct Llm {
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

/// Represents a team of collaborating LLMs, which can be a single LLM, a vertical collaboration, or a horizontal collaboration.
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

/// Represents the genome of an LLMTeam in the genetic algorithm framework.
#[derive(Clone)]
struct LlmTeamGenome {
    dna: Vec<LLMGene>,
    age: i32,
    tasks: Vec<Task>,
}

impl Default for LlmTeamGenome {
    fn default() -> Self {
        Self {
            dna: Default::default(),
            age: Default::default(),
            tasks: Task::new(1000),
        }
    }
}

impl LlmTeamGenome {
    fn parse_genotype(&self) -> LlmTeam {
        if self.dna.is_empty() {
            return LlmTeam::Invalid;
        }

        let mut team_stack: Vec<LlmTeam> = Vec::new();

        for gene in self.dna.iter().rev() {
            match gene {
                LLMGene::Single => team_stack.push(LlmTeam::Single(Llm::default())),
                LLMGene::Horizontal => {
                    let mut members: Vec<LlmTeam> = Vec::new();
                    while let Some(team) = team_stack.pop() {
                        if let LlmTeam::Vertical { .. } = team {
                            team_stack.push(team);
                            break;
                        }
                        members.push(team);
                    }
                    team_stack.push(LlmTeam::Horizontal { members });
                }
                LLMGene::Vertical => {
                    let mut followers: Vec<LlmTeam> = Vec::new();
                    while let Some(team) = team_stack.pop() {
                        if let LlmTeam::Horizontal { .. } = team {
                            team_stack.push(team);
                            break;
                        }
                        followers.push(team);
                    }
                    if let Some(leader) = team_stack.pop() {
                        team_stack.push(LlmTeam::Vertical {
                            leader: Box::new(leader),
                            followers,
                        });
                    } else {
                        return LlmTeam::Invalid;
                    }
                }
            }
        }

        if team_stack.len() == 1 {
            team_stack.pop().unwrap()
        } else {
            LlmTeam::Invalid
        }
    }
}

/// Represents a task that the LLMTeams need to solve.
#[derive(Clone)]
struct Task {
    root_task: SubTask,
    subtasks: Vec<SubTask>,
    economic_value: f32,
}

#[derive(Clone)]
/// Represents a subtask that is part of a larger task.
struct SubTask {
    id: String,
    description: String,
    reasoning_required: f32,
    planning_required: f32,
    competency_required: f32,
    token_estimate: usize,
}

impl Default for SubTask {
    fn default() -> Self {
        let mut rng = SmallRng::from_entropy();

        // Normal distribution for difficulty-related fields
        let difficulty_dist = Normal::new(TASK_DIFFICULTY_MEAN, TASK_DIFFICULTY_STD_DEV).unwrap();

        Self {
            id: Default::default(),
            description: Default::default(),
            reasoning_required: difficulty_dist.sample(&mut rng).clamp(0.0, 1.0) as f32,
            planning_required: difficulty_dist.sample(&mut rng).clamp(0.0, 1.0) as f32,
            competency_required: difficulty_dist.sample(&mut rng).clamp(0.0, 1.0) as f32,
            token_estimate: Uniform::new(SUBTASK_TOKEN_ESTIMATE_MIN, SUBTASK_TOKEN_ESTIMATE_MAX)
                .sample(&mut rng),
        }
    }
}

impl Default for Task {
    fn default() -> Self {
        let mut rng = SmallRng::from_entropy();
        let root_task_token_estimate = Normal::new(
            ROOT_TASK_TOKEN_ESTIMATE_MEAN,
            ROOT_TASK_TOKEN_ESTIMATE_STD_DEV,
        )
        .unwrap()
        .sample(&mut rng)
        .clamp(
            SUBTASK_TOKEN_ESTIMATE_MIN as f64,
            SUBTASK_TOKEN_ESTIMATE_MAX as f64,
        ) as usize;
        let root_task = SubTask {
            token_estimate: root_task_token_estimate,
            ..Default::default()
        };

        let num_subtasks = Uniform::new(SUBTASK_COUNT_MIN, SUBTASK_COUNT_MAX).sample(&mut rng);
        let avg_subtask_token_estimate = root_task_token_estimate as f32 / num_subtasks as f32;

        let subtasks = (0..num_subtasks)
            .map(|_| {
                let token_estimate = Normal::new(
                    avg_subtask_token_estimate,
                    ROOT_TASK_TOKEN_ESTIMATE_STD_DEV as f32,
                )
                .unwrap()
                .sample(&mut rng)
                .clamp(
                    SUBTASK_TOKEN_ESTIMATE_MIN as f32,
                    SUBTASK_TOKEN_ESTIMATE_MAX as f32,
                ) as usize;
                SubTask {
                    token_estimate,
                    ..Default::default()
                }
            })
            .collect();

        let avg_difficulty = (root_task.reasoning_required
            + root_task.planning_required
            + root_task.competency_required)
            / 3.0;
        let economic_value = (1.0 - avg_difficulty) * root_task_token_estimate as f32;

        Self {
            root_task,
            subtasks,
            economic_value,
        }
    }
}

impl Task {
    fn new(amount: usize) -> Vec<Self> {
        (0..amount).map(|_| Task::default()).collect()
    }
}

impl GenotypeT for LlmTeamGenome {
    type Gene = LLMGene;

    fn get_dna(&self) -> &[Self::Gene] {
        &self.dna
    }

    fn set_dna(&mut self, dna: &[Self::Gene]) {
        self.dna = dna.to_vec();
    }

    /// Calculates the fitness of the LLMTeamGenome based on its performance on the tasks.
    fn calculate_fitness(&mut self) {
        unimplemented!("Calculate fitness based on task performance");
    }

    fn get_fitness(&self) -> f64 {
        todo!()
    }

    fn set_fitness(&mut self, fitness: f64) {
        todo!()
    }

    fn set_age(&mut self, age: i32) {
        self.age = age;
    }

    fn get_age(&self) -> i32 {
        self.age
    }
}

fn main() {
    // Configure and run the genetic algorithm
    let mut population: Population<LlmTeamGenome> = ga::Ga::new()
        .with_threads(8)
        .with_problem_solving(ProblemSolving::Maximization)
        .with_selection_method(Selection::Tournament)
        .with_number_of_couples(10)
        .with_crossover_method(Crossover::Cycle)
        .with_mutation_method(Mutation::Swap)
        .with_survivor_method(Survivor::Fitness)
        .with_genes_per_individual(100) // Adjust based on expected genotype length
        .with_population_size(100)
        .run();

    // Sort the individuals in the population by their fitness
    population
        .individuals
        .sort_by(|a, b| b.get_fitness().partial_cmp(&a.get_fitness()).unwrap());

    // Get the best individual from the sorted population
    let best_genome = &population.individuals[0];

    // Parse the genotype of the best individual
    let best_team = best_genome.parse_genotype();

    // Evaluate the performance of the best team on the tasks
    // ...

    // Print the results
    // ...
}
