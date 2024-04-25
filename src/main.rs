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

use genetic_algorithms::traits::{GeneT, GenotypeT};

const GENOME_LENGTH: usize = 3 * 2;
const LLM_TEAMS_PER_TASK: usize = 10;

// Constants for task and subtask generation
const TASK_DIFFICULTY_MEAN: f64 = 0.8;
const TASK_DIFFICULTY_STD_DEV: f64 = 0.2;
const SUBTASK_TOKEN_ESTIMATE_MIN: usize = 1000;
const SUBTASK_TOKEN_ESTIMATE_MAX: usize = 200000;
const ROOT_TASK_TOKEN_ESTIMATE_MEAN: f64 = 24000.0;
const ROOT_TASK_TOKEN_ESTIMATE_STD_DEV: f64 = 15000.0;
const SUBTASK_COUNT_MIN: usize = 1;
const SUBTASK_COUNT_MAX: usize = 15;

// Constants for LLM generation
const LLM_CONTEXT_LENGTH_MIN: usize = 4096;
const LLM_CONTEXT_LENGTH_MAX: usize = 262144;
const LLM_CONTEXT_LENGTH_MEAN: f64 = 8192.0;
const LLM_CONTEXT_LENGTH_STD_DEV: f64 = 4096.0;
const LLM_ABILITY_MEAN: f64 = 0.7;
const LLM_ABILITY_STD_DEV: f64 = 0.3;
const LLM_VERBOSITY_MIN: f64 = 0.5;
const LLM_VERBOSITY_MAX: f64 = 1.5;
const LLM_COST_PER_TOKEN_MIN: f64 = 0.0000006;
const LLM_COST_PER_TOKEN_MAX: f64 = 0.00003;
const LLM_COST_PER_TOKEN_MEAN: f64 = 0.0000015;
const LLM_COST_PER_TOKEN_STD_DEV: f64 = 0.000001;

/// Represents the different types of genes in the LLMTeamGenome.
#[derive(Clone, Copy, Debug)]
enum LLMGene {
    /// Represents a single LLM instance in the team, working independently.
    Single(i32),
    /// Represents a vertical collaboration structure with a leader and followers.
    Vertical(i32),
    /// Represents a horizontal collaboration structure with multiple members.
    Horizontal(i32),
}

impl Default for LLMGene {
    fn default() -> Self {
        LLMGene::Single(0)
    }
}
impl LLMGene {
    fn from_id(id: i32) -> Self {
        match id % 3 {
            0 => LLMGene::Single(id as i32),
            1 => LLMGene::Vertical(id as i32),
            2 => LLMGene::Horizontal(id as i32),
            _ => panic!("Invalid gene ID"),
        }
    }
}

impl GeneT for LLMGene {
    fn set_id(&mut self, id: i32) {
        *self = match id.abs() % 3 {
            0 => LLMGene::Single(id),
            1 => LLMGene::Vertical(id),
            2 => LLMGene::Horizontal(id),
            _ => unreachable!(),
        };
    }

    fn get_id(&self) -> i32 {
        match self {
            LLMGene::Single(i) => *i,
            LLMGene::Vertical(i) => *i,
            LLMGene::Horizontal(i) => *i,
        }
    }
}

/// Represents an individual language model with its specific characteristics and capabilities.
#[derive(Clone, Copy, Debug)]
struct Llm {
    context_length: usize,
    planning_ability: f32,
    verbosity: f32,
    competency: f32,
    cost_per_token: f32,
}

impl Default for Llm {
    fn default() -> Self {
        let mut rng = SmallRng::from_entropy();

        // Sample the context length as a power of 2
        let context_length_power = Normal::new(
            LLM_CONTEXT_LENGTH_MEAN.log2(),
            LLM_CONTEXT_LENGTH_STD_DEV.log2(),
        )
        .unwrap()
        .sample(&mut rng)
        .clamp(
            (LLM_CONTEXT_LENGTH_MIN as f64).log2(),
            (LLM_CONTEXT_LENGTH_MAX as f64).log2(),
        );
        let context_length = 2_usize.pow(context_length_power as u32);

        // Sample the reasoning, planning, and competency abilities
        let ability_dist =
            Normal::new(LLM_ABILITY_MEAN, LLM_ABILITY_STD_DEV).expect("Invalid dist.");
        let reasoning_ability = ability_dist.sample(&mut rng).clamp(0.0, 1.0) as f32;
        let planning_ability = ability_dist.sample(&mut rng).clamp(0.0, 1.0) as f32;
        let competency_1 = ability_dist.sample(&mut rng).clamp(0.0, 1.0) as f32;
        let competency_2 = ability_dist.sample(&mut rng).clamp(0.0, 1.0) as f32;
        let competency_3 = ability_dist.sample(&mut rng).clamp(0.0, 1.0) as f32;
        // Calculate the average competency
        let avg_competency =
            (reasoning_ability + planning_ability + competency_1 + competency_2 + competency_3)
                / 5.0;

        // Sample the verbosity based on the average competency
        let verbosity = Normal::new(avg_competency as f64, 0.15)
            .unwrap()
            .sample(&mut rng)
            .clamp(LLM_VERBOSITY_MIN, LLM_VERBOSITY_MAX) as f32;

        // Sample the cost per output token based on the average competency
        let cost_per_output_token = Normal::new(
            LLM_COST_PER_TOKEN_MEAN + (avg_competency as f64 - LLM_ABILITY_MEAN) * 0.000001,
            LLM_COST_PER_TOKEN_STD_DEV,
        )
        .unwrap()
        .sample(&mut rng)
        .clamp(LLM_COST_PER_TOKEN_MIN, LLM_COST_PER_TOKEN_MAX);

        // Calculate the cost per input token
        let cost_per_input_token = cost_per_output_token / 5.0;
        let cost_per_token = (0.75 * cost_per_input_token + 0.25 * cost_per_output_token) as f32;
        let competency = (competency_1 + competency_2 + competency_3) / 3.0;
        let planning_ability = (planning_ability + reasoning_ability) / 2.0;
        Self {
            context_length,
            planning_ability,
            verbosity,
            competency,
            cost_per_token,
        }
    }
}

#[derive(Debug)]
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
#[derive(Clone, Default, Debug)]
struct LlmTeamGenome {
    dna: [LLMGene; GENOME_LENGTH],
    age: i32,
    fitness: f64,
}

impl LlmTeamGenome {
    fn parse_genotype(&self, llms: &[Llm]) -> LlmTeam {
        if self.dna.is_empty() {
            return LlmTeam::Invalid;
        }

        // Sort the LLMs by their average competency and planning ability in descending order
        let mut sorted_llms = llms.to_vec();
        sorted_llms.sort_by(|a, b| {
            let score_a = (a.competency + a.planning_ability) / 2.0;
            let score_b = (b.competency + b.planning_ability) / 2.0;

            score_b.partial_cmp(&score_a).unwrap()
        });

        let mut team_stack: Vec<LlmTeam> = Vec::new();
        let mut llm_iter = sorted_llms.iter();

        match self.dna[0] {
            LLMGene::Single(_) => {
                if let Some(llm) = llm_iter.next() {
                    team_stack.push(LlmTeam::Single(*llm));
                } else {
                    return LlmTeam::Invalid;
                }
            }
            _ => {
                for gene in self.dna.iter().skip(1).rev() {
                    match gene {
                        LLMGene::Single(_) => {
                            if let Some(llm) = llm_iter.next() {
                                team_stack.push(LlmTeam::Single(*llm));
                            } else {
                                break;
                            }
                        }
                        LLMGene::Vertical(_) => {
                            let mut followers: Vec<LlmTeam> = Vec::new();
                            while let Some(team) = team_stack.pop() {
                                followers.push(team);
                            }
                            team_stack.push(LlmTeam::Vertical {
                                leader: Box::new(LlmTeam::Single(*llm_iter.next().unwrap())),
                                followers,
                            });
                        }
                        LLMGene::Horizontal(_) => {
                            let mut followers: Vec<LlmTeam> = Vec::new();
                            while let Some(team) = team_stack.pop() {
                                followers.push(team);
                            }
                            team_stack.push(LlmTeam::Horizontal { members: followers });
                        }
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
    llms: Vec<Vec<Llm>>,
}

#[derive(Clone)]
/// Represents a subtask that is part of a larger task.
struct SubTask {
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
            llms: vec![vec![Llm::default(); GENOME_LENGTH]; LLM_TEAMS_PER_TASK],
        }
    }
}

impl Task {
    fn new(amount: usize) -> Vec<Self> {
        (0..amount).map(|_| Task::default()).collect()
    }
}

impl LlmTeam {
    fn solve_task(&self, task: &Task) -> bool {
        match self {
            LlmTeam::Single(llm) => llm.solve_task(task),
            LlmTeam::Vertical { leader, followers } => {
                let subtasks = leader.break_down_task(task);
                if subtasks.is_empty() {
                    return false;
                }
                if followers.is_empty() {
                    return false;
                }
                let chunks = subtasks
                    .chunks(followers.len())
                    .map(|c| c.to_vec())
                    .collect::<Vec<_>>();
                followers.iter().zip(chunks).all(|(follower, subtasks)| {
                    subtasks.iter().all(|subtask| {
                        follower.solve_task(&Task {
                            root_task: subtask.clone(),
                            subtasks: vec![],
                            economic_value: 0.0,
                            llms: vec![],
                        })
                    })
                })
            }
            LlmTeam::Horizontal { members } => {
                let results = members
                    .iter()
                    .map(|member| member.solve_task(task))
                    .collect::<Vec<_>>();
                results.iter().filter(|&&r| r).count() > results.len() / 2
            }
            LlmTeam::Invalid => false,
        }
    }

    fn break_down_task(&self, task: &Task) -> Vec<SubTask> {
        if let LlmTeam::Single(llm) = self {
            llm.break_down_task(task)
        } else {
            vec![]
        }
    }
}

impl Llm {
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
}

impl GenotypeT for LlmTeamGenome {
    type Gene = LLMGene;

    fn get_dna(&self) -> &[Self::Gene] {
        &self.dna
    }

    fn set_dna(&mut self, dna: &[Self::Gene]) {
        self.dna = dna.try_into().expect("Invalid DNA length");
    }

    /// Calculates the fitness of the LLMTeamGenome based on its performance on the tasks.
    fn calculate_fitness(&mut self) {
        let mut total_fitness = 0.0;
        let tasks = Task::new(10);

        for task in tasks {
            for llm_team in &task.llms {
                let llm_team = self.parse_genotype(llm_team);
                let success_count = (0..LLM_TEAMS_PER_TASK)
                    .filter(|_| llm_team.solve_task(&task))
                    .count();
                total_fitness += success_count as f64 / LLM_TEAMS_PER_TASK as f64;
            }
        }
        self.fitness += total_fitness;

        // flip a coin, 50% chance of success
        // let mut rng = thread_rng();
        // let success = rng.gen_bool(0.5);
        // self.fitness += if success { 1.0 } else { 0.0 };
    }

    fn get_fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn set_age(&mut self, age: i32) {
        self.age = age;
    }

    fn get_age(&self) -> i32 {
        self.age
    }
}

fn generate_all_combinations() -> Vec<LlmTeamGenome> {
    let mut combinations = Vec::new();
    for i in 0..3_i32.pow(GENOME_LENGTH as u32) {
        let mut genome = LlmTeamGenome {
            dna: [LLMGene::Single(0); GENOME_LENGTH],
            age: 0,
            fitness: 0.0,
        };
        let mut genes = (0..GENOME_LENGTH)
            .map(|j| LLMGene::from_id((i + j as i32) % GENOME_LENGTH as i32))
            .collect::<Vec<_>>();
        // shuffle the genes
        let mut rng = thread_rng();
        genes.shuffle(&mut rng);
        genome.set_dna(&genes);
        combinations.push(genome.clone());
        genes.shuffle(&mut rng);
        genome.set_dna(&genes);
        combinations.push(genome);
    }
    combinations
}

fn main() {
    let individuals = generate_all_combinations();

    let mut population = Population::new(individuals);
    println!("Population size: {}", population.individuals.len());

    population = ga::Ga::new()
        .with_threads(num_cpus::get() as i32)
        .with_problem_solving(ProblemSolving::Maximization)
        .with_selection_method(Selection::RouletteWheel)
        .with_number_of_couples(2)
        .with_crossover_method(Crossover::Cycle)
        .with_mutation_method(Mutation::Swap)
        .with_survivor_method(Survivor::Fitness)
        // .with_logs(genetic_algorithms::configuration::LogLevel::Trace)
        .with_alleles(vec![
            LLMGene::from_id(0),
            LLMGene::from_id(1),
            LLMGene::from_id(2),
            LLMGene::from_id(3),
            LLMGene::from_id(4),
            LLMGene::from_id(5),
        ])
        .with_adaptive_ga(true)
        .with_crossover_probability_min(0.2)
        .with_crossover_probability_max(0.8)
        .with_population(population)
        .with_max_generations(1000)
        .run();

    // Sort the individuals in the population by their fitness
    population
        .individuals
        .sort_by(|a, b| b.get_fitness().partial_cmp(&a.get_fitness()).unwrap());

    println!("Best individual: {:?}", population.individuals);
}
