using GeneticSharp;
using GeneticSharp.Extensions;
using MathNet.Numerics.Distributions;
using LanguageExt;
using System.Diagnostics;
using System.Collections.Generic;

public class Llm
{
    public int ContextLength { get; }
    public float PlanningAbility { get; }
    public float Verbosity { get; }
    public float Competency { get; }
    public float CostPerToken { get; }

    private static readonly Normal AbilityDistribution = new Normal(0.7, 0.3);
    private static readonly LogNormal ContextLengthDistribution = new LogNormal(Math.Log(8192), Math.Log(2));
    private static readonly ContinuousUniform VerbosityDistribution = new ContinuousUniform(0.5, 1.5);
    private static readonly LogNormal CostPerTokenDistribution = new LogNormal(Math.Log(0.0000015), Math.Log(1.5));

    public Llm()
    {
        ContextLength = (int)Math.Clamp(Math.Pow(2, Math.Round(ContextLengthDistribution.Sample())), 4096, 262144);
        PlanningAbility = (float)Math.Clamp(AbilityDistribution.Sample(), 0, 1);
        Competency = (float)Math.Clamp(AbilityDistribution.Sample(), 0, 1);
        Verbosity = (float)VerbosityDistribution.Sample();
        CostPerToken = (float)Math.Clamp(CostPerTokenDistribution.Sample(), 0.0000006, 0.00003);
    }

    public bool SolveTask(Task task)
    {
        float scaledCompetency = Competency * task.RootTask.CompetencyRequired;
        float scaledPlanningAbility = PlanningAbility * task.RootTask.PlanningRequired;
        float avgCompetency = (scaledCompetency + scaledPlanningAbility) / 2f;

        int scaledTokenEstimate = (int)(task.RootTask.TokenEstimate * Verbosity);
        float adjustedCompetency = scaledTokenEstimate > ContextLength ? avgCompetency / 2f : avgCompetency;

        return new Normal(adjustedCompetency, 0.2).Sample() > 0.5;
    }

    public List<SubTask> BreakDownTask(Task task)
    {
        float threshold = 1f - (task.RootTask.ReasoningRequired + task.RootTask.PlanningRequired) / 2f;
        bool success = new Normal(PlanningAbility, 0.1).Sample() > threshold;

        if (success)
        {
            return task.Subtasks;
        }
        else
        {
            return task.Subtasks.Select(subtask => new SubTask
            {
                ReasoningRequired = Math.Min(subtask.ReasoningRequired + 0.1f, 1f),
                PlanningRequired = Math.Min(subtask.PlanningRequired + 0.1f, 1f),
                CompetencyRequired = Math.Min(subtask.CompetencyRequired + 0.1f, 1f),
                TokenEstimate = subtask.TokenEstimate
            }).ToList();
        }
    }
}

public enum LLMGene
{
    Single = 0,
    Vertical = 1,
    Horizontal = 2
}
public abstract class LlmTeam
{
    public abstract bool SolveTask(Task task);
    public abstract List<SubTask> BreakDownTask(Task task);
}

public class SingleLlmTeam : LlmTeam
{
    public Llm Llm { get; }

    public SingleLlmTeam(Llm llm)
    {
        Llm = llm;
    }

    public override bool SolveTask(Task task) => Llm.SolveTask(task);

    public override List<SubTask> BreakDownTask(Task task) => Llm.BreakDownTask(task);
}

public class VerticalLlmTeam : LlmTeam
{
    public LlmTeam Leader { get; }
    public List<LlmTeam> Followers { get; }

    public VerticalLlmTeam(LlmTeam leader, List<LlmTeam> followers)
    {
        Leader = leader;
        Followers = followers;
    }

    public override bool SolveTask(Task task)
    {
        var subtasks = Leader.BreakDownTask(task);
        if (subtasks.Count == 0 || Followers.Count == 0)
        {
            return false;
        }

        var chunks = subtasks.Chunk(Followers.Count).ToList();
        return Followers.Zip(chunks).All(pair =>
        {
            var (follower, subtaskChunk) = pair;
            return subtaskChunk.All(subtask =>
                follower.SolveTask(new Task())
            );
        });
    }

    public override List<SubTask> BreakDownTask(Task task) => new List<SubTask>();
}

public class HorizontalLlmTeam : LlmTeam
{
    public List<LlmTeam> Members { get; }

    public HorizontalLlmTeam(List<LlmTeam> members)
    {
        Members = members;
    }

    public override bool SolveTask(Task task)
    {
        var results = Members.Select(member => member.SolveTask(task)).ToList();
        return results.Count(r => r) > results.Count / 2;
    }

    public override List<SubTask> BreakDownTask(Task task) => new List<SubTask>();
}

public class LlmTeamChromosome : ChromosomeBase
{
    public int Age { get; set; }
    public double FitnessValue { get; set; }

    private const int GenomeLength = 6;

    public LlmTeamChromosome() : base(GenomeLength)
    {
        CreateGenes();
    }

    public override Gene GenerateGene(int geneIndex)
    {
        return new Gene(RandomizationProvider.Current.GetInt(0, Enum.GetValues(typeof(LLMGene)).Length));
    }

    public override IChromosome CreateNew()
    {
        return new LlmTeamChromosome();
    }

    public override IChromosome Clone()
    {
        var clone = base.Clone() as LlmTeamChromosome;
        if (clone == null)
        {
            throw new InvalidOperationException("Failed to clone chromosome.");
        }
        clone.Age = this.Age;
        clone.FitnessValue = this.FitnessValue;
        return clone;
    }

    public List<LLMGene> GetDna()
    {
        return GetGenes().Select(g => (LLMGene)(int)g.Value).ToList();
    }

    public Option<LlmTeam> ParseGenotype(List<Llm> llms)
    {
        var dna = GetDna();
        if (dna == null || dna.Count == 0)
        {
            return Option<LlmTeam>.None;
        }

        var sortedLlms = llms.OrderByDescending(llm => (llm.Competency + llm.PlanningAbility) / 2).ToList();
        var teamStack = new Stack<LlmTeam>();
        var llmIter = sortedLlms.GetEnumerator();

        if (dna[0] == LLMGene.Single)
        {
            if (llmIter.MoveNext())
            {
                return Option<LlmTeam>.Some(new SingleLlmTeam(llmIter.Current));
            }
            return Option<LlmTeam>.None;
        }

        for (int i = dna.Count - 1; i >= 1; i--)
        {
            switch (dna[i])
            {
                case LLMGene.Single:
                    if (llmIter.MoveNext())
                    {
                        teamStack.Push(new SingleLlmTeam(llmIter.Current));
                    }
                    break;
                case LLMGene.Vertical:
                    if (llmIter.MoveNext() && teamStack.Count > 0)
                    {
                        var followers = new List<LlmTeam>();
                        while (teamStack.Count > 0)
                        {
                            followers.Add(teamStack.Pop());
                        }
                        teamStack.Push(new VerticalLlmTeam(new SingleLlmTeam(llmIter.Current), followers));
                    }
                    break;
                case LLMGene.Horizontal:
                    if (teamStack.Count > 0)
                    {
                        var members = new List<LlmTeam>();
                        while (teamStack.Count > 0)
                        {
                            members.Add(teamStack.Pop());
                        }
                        teamStack.Push(new HorizontalLlmTeam(members));
                    }
                    break;
            }
        }

        if (teamStack.Count == 1)
        {
            return Option<LlmTeam>.Some(teamStack.Pop());
        }
        else if (teamStack.Count > 1)
        {
            return Option<LlmTeam>.Some(new HorizontalLlmTeam(teamStack.ToList()));
        }
        else
        {
            return Option<LlmTeam>.None;
        }
    }
}
public class LlmTeamFitness : IFitness
{
    private const int TaskCount = 10;
    private const int LlmTeamsPerTask = 100;

    public double Evaluate(IChromosome chromosome)
    {
        var genome = chromosome as LlmTeamChromosome;
        if (genome == null)
        {
            return 0;
        }

        double totalFitness = 0.0;
        var tasks = Task.GenerateTasks(TaskCount);

        foreach (var task in tasks)
        {
            for (int i = 0; i < LlmTeamsPerTask; i++)
            {
                var llmTeamOption = genome.ParseGenotype(task.Llms[i]);

                llmTeamOption.Match(
                    Some: llmTeam =>
                    {
                        if (llmTeam.SolveTask(task))
                        {
                            totalFitness += 1.0 / LlmTeamsPerTask;
                        }
                    },
                    None: () => { /* Invalid team, do nothing */ }
                );
            }
        }

        genome.FitnessValue = totalFitness;
        return totalFitness;
    }
}

public class SubTask
{
    private static readonly Normal DifficultyDistribution = new Normal(0.8, 0.2);
    private static readonly ContinuousUniform TokenEstimateDistribution = new ContinuousUniform(1000, 200000);

    public float ReasoningRequired { get; set; }
    public float PlanningRequired { get; set; }
    public float CompetencyRequired { get; set; }
    public int TokenEstimate { get; set; }

    public SubTask()
    {
        ReasoningRequired = (float)Math.Clamp(DifficultyDistribution.Sample(), 0, 1);
        PlanningRequired = (float)Math.Clamp(DifficultyDistribution.Sample(), 0, 1);
        CompetencyRequired = (float)Math.Clamp(DifficultyDistribution.Sample(), 0, 1);
        TokenEstimate = (int)TokenEstimateDistribution.Sample();
    }

    public SubTask Clone() => (SubTask)MemberwiseClone();
}
public class Task
{
    private static readonly Normal RootTaskTokenEstimateDistribution = new Normal(24000, 15000);
    private static readonly ContinuousUniform SubtaskCountDistribution = new ContinuousUniform(1, 15);

    public SubTask RootTask { get; set; }
    public List<SubTask> Subtasks { get; set; }
    public float EconomicValue { get; set; }
    public List<List<Llm>> Llms { get; set; }

    public Task()
    {
        RootTask = new SubTask
        {
            TokenEstimate = (int)Math.Clamp(RootTaskTokenEstimateDistribution.Sample(), 1000, 200000)
        };

        int numSubtasks = (int)SubtaskCountDistribution.Sample();
        float avgSubtaskTokenEstimate = RootTask.TokenEstimate / (float)numSubtasks;

        Subtasks = Enumerable.Range(0, numSubtasks)
            .Select(_ => new SubTask
            {
                TokenEstimate = (int)Math.Clamp(new Normal(avgSubtaskTokenEstimate, 15000).Sample(), 1000, 200000)
            })
            .ToList();

        float avgDifficulty = (RootTask.ReasoningRequired + RootTask.PlanningRequired + RootTask.CompetencyRequired) / 3f;
        EconomicValue = (1f - avgDifficulty) * RootTask.TokenEstimate;

        Llms = Enumerable.Range(0, 100)
            .Select(_ => Enumerable.Range(0, 6).Select(_ => new Llm()).ToList())
            .ToList();
    }

    public static List<Task> GenerateTasks(int amount) =>
        Enumerable.Range(0, amount).Select(_ => new Task()).ToList();
}


class Program
{
    static void PrintTeamStructure(LlmTeam team, string indent = "")
    {
        switch (team)
        {
            case SingleLlmTeam singleTeam:
                Console.WriteLine($"{indent}Single LLM (Competency: {singleTeam.Llm.Competency:F2}, Planning Ability: {singleTeam.Llm.PlanningAbility:F2})");
                break;
            case VerticalLlmTeam verticalTeam:
                Console.WriteLine($"{indent}Vertical LLM Team:");
                Console.WriteLine($"{indent}  Leader:");
                PrintTeamStructure(verticalTeam.Leader, indent + "    ");
                Console.WriteLine($"{indent}  Followers:");
                foreach (var follower in verticalTeam.Followers)
                {
                    PrintTeamStructure(follower, indent + "    ");
                }
                break;
            case HorizontalLlmTeam horizontalTeam:
                Console.WriteLine($"{indent}Horizontal LLM Team:");
                foreach (var member in horizontalTeam.Members)
                {
                    PrintTeamStructure(member, indent + "  ");
                }
                break;
            default:
                Console.WriteLine($"{indent}Unknown team type.");
                break;
        }
    }
    static void Main(string[] args)
    {
        int maxGenerations = 10;
        bool useAutoConfig = false;

        // Parse command-line arguments
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-m" && i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedValue))
            {
                maxGenerations = parsedValue;
                i++; // Skip next argument as it's part of this option
            }
            else if (args[i] == "--autoconfig")
            {
                useAutoConfig = true;
            }
        }

        if (useAutoConfig)
        {
            RunAutoConfigGA(maxGenerations);
        }
        else
        {
            RunStandardGA(maxGenerations);
        }
    }

    static void AnalyzePopulation(Population population, int generationNumber)
    {
        int singleTeams = 0;
        int verticalTeams = 0;
        int horizontalTeams = 0;
        int hybridTeams = 0;
        int invalidTeams = 0;

        foreach (var chromosome in population.CurrentGeneration.Chromosomes)
        {
            var llmChromosome = chromosome as LlmTeamChromosome;
            if (llmChromosome == null) continue;

            // Generate LLMs and parse the genotype
            var sampleLlms = Task.GenerateTasks(1)[0].Llms[0];
            var teamOption = llmChromosome.ParseGenotype(sampleLlms);

            teamOption.Match(
                Some: team =>
                {
                    var teamType = GetTeamType(team);
                    switch (teamType)
                    {
                        case "Single":
                            singleTeams++;
                            break;
                        case "Vertical":
                            verticalTeams++;
                            break;
                        case "Horizontal":
                            horizontalTeams++;
                            break;
                        case "Hybrid":
                            hybridTeams++;
                            break;
                    }
                },
                None: () => invalidTeams++
            );
        }

        Console.WriteLine($"Generation {generationNumber} Population Statistics:");
        Console.WriteLine($"Single Teams: {singleTeams}");
        Console.WriteLine($"Vertical Teams: {verticalTeams}");
        Console.WriteLine($"Horizontal Teams: {horizontalTeams}");
        Console.WriteLine($"Hybrid Teams: {hybridTeams}");
        Console.WriteLine($"Invalid Teams: {invalidTeams}");
        Console.WriteLine();
    }

    static string GetTeamType(LlmTeam team)
    {
        bool hasVertical = false;
        bool hasHorizontal = false;

        void TraverseTeam(LlmTeam t)
        {
            switch (t)
            {
                case SingleLlmTeam _:
                    break;
                case VerticalLlmTeam verticalTeam:
                    hasVertical = true;
                    TraverseTeam(verticalTeam.Leader);
                    foreach (var follower in verticalTeam.Followers)
                    {
                        TraverseTeam(follower);
                    }
                    break;
                case HorizontalLlmTeam horizontalTeam:
                    hasHorizontal = true;
                    foreach (var member in horizontalTeam.Members)
                    {
                        TraverseTeam(member);
                    }
                    break;
            }
        }

        TraverseTeam(team);

        if (hasVertical && hasHorizontal)
            return "Hybrid";
        if (hasVertical)
            return "Vertical";
        if (hasHorizontal)
            return "Horizontal";
        return "Single";
    }


    static void RunStandardGA(int maxGenerations)
    {
        var selection = new TournamentSelection();
        var crossover = new OnePointCrossover();
        var mutation = new TworsMutation();
        var fitness = new LlmTeamFitness();
        var chromosome = new LlmTeamChromosome();
        var population = new Population(100, 200, chromosome);

        var ga = new GeneticAlgorithm(population, fitness, selection, crossover, mutation)
        {
            Termination = new GenerationNumberTermination(maxGenerations),
            MutationProbability = 0.2f,
            CrossoverProbability = 0.8f,
            TaskExecutor = new ParallelTaskExecutor(),
        };

        // Implementing Elitism
        // ga.Reinsertion = new ElitistReinsertion();

        // Logging GA progress
        ga.GenerationRan += (sender, e) =>
        {
            var bestFitness = ga.BestChromosome.Fitness;
            var bestChromosome = ga.BestChromosome as LlmTeamChromosome;
            Console.WriteLine($"Generation {ga.GenerationsNumber}: Best Fitness = {bestFitness}");

            if (bestChromosome != null)
            {
                Console.WriteLine($"Best genome: {string.Join(", ", bestChromosome.GetDna())}");

                // Use a sample task to parse and display the team structure
                var sampleLlms = Task.GenerateTasks(1)[0].Llms[0];
                var bestTeamOption = bestChromosome.ParseGenotype(sampleLlms);

                bestTeamOption.Match(
                    Some: bestTeam =>
                    {
                        Console.WriteLine("Best team structure:");
                        PrintTeamStructure(bestTeam);
                    },
                    None: () => Console.WriteLine("Invalid team structure.")
                );
            }
        };


        Console.WriteLine("Starting genetic algorithm...");
        ga.Start();

        Console.WriteLine($"Best solution found has {ga.BestChromosome.Fitness} fitness.");
        var bestChromosome = ga.BestChromosome as LlmTeamChromosome;
        if (bestChromosome != null)
        {
            Console.WriteLine($"Best genome: {string.Join(", ", bestChromosome.GetDna())}");

            // Assuming you have a list of LLMs to use (e.g., from a task)
            var sampleLlms = Task.GenerateTasks(1)[0].Llms[0];
            var bestTeamOption = bestChromosome.ParseGenotype(sampleLlms);

            bestTeamOption.Match(
                Some: bestTeam =>
                {
                    Console.WriteLine("Winning team structure:");
                    PrintTeamStructure(bestTeam);
                },
                None: () => Console.WriteLine("Invalid team structure.")
            );
        }
    }

    static void RunAutoConfigGA(int maxGenerations)
    {
        Console.WriteLine("Starting AutoConfig genetic algorithm to optimize GA operators...");

        var targetFitness = new LlmTeamFitness();
        var targetChromosome = new LlmTeamChromosome();

        // Create a custom AutoConfigChromosome with only compatible operators
        var autoConfigFitness = new AutoConfigFitness(targetFitness, targetChromosome)
        {
            PopulationMinSize = 50,
            PopulationMaxSize = 100,
            Termination = new GenerationNumberTermination(maxGenerations / 2),
            TaskExecutor = new ParallelTaskExecutor(),
        };

        var autoConfigChromosome = new CustomAutoConfigChromosome();
        var autoConfigPopulation = new Population(20, 40, autoConfigChromosome);

        var autoConfigGa = new GeneticAlgorithm(autoConfigPopulation, autoConfigFitness, new EliteSelection(), new UniformCrossover(), new UniformMutation())
        {
            Termination = new GenerationNumberTermination(maxGenerations / 2),
            MutationProbability = 0.3f,
            CrossoverProbability = 0.7f,
            TaskExecutor = new ParallelTaskExecutor(),
        };

        autoConfigGa.GenerationRan += (sender, e) =>
        {
            var bestAutoChromosome = autoConfigGa.BestChromosome as CustomAutoConfigChromosome;
            Console.WriteLine($"Meta Generation {autoConfigGa.GenerationsNumber}: Best Meta Fitness = {bestAutoChromosome?.Fitness}");
        };

        autoConfigGa.Start();

        var bestAutoConfigChromosome = autoConfigGa.BestChromosome as CustomAutoConfigChromosome;

        if (bestAutoConfigChromosome != null)
        {
            // Extract the optimized operators
            var selection = bestAutoConfigChromosome.Selection;
            var crossover = bestAutoConfigChromosome.Crossover;
            var mutation = bestAutoConfigChromosome.Mutation;

            Console.WriteLine("Optimized GA operators found:");
            Console.WriteLine($"Selection Operator: {selection.GetType().Name}");
            Console.WriteLine($"Crossover Operator: {crossover.GetType().Name}");
            Console.WriteLine($"Mutation Operator: {mutation.GetType().Name}");

            // Run the standard GA with optimized operators
            var fitness = new LlmTeamFitness();
            var chromosome = new LlmTeamChromosome();
            var population = new Population(100, 200, chromosome);

            var ga = new GeneticAlgorithm(population, fitness, selection, crossover, mutation)
            {
                Termination = new GenerationNumberTermination(maxGenerations),
                MutationProbability = 0.2f,
                CrossoverProbability = 0.8f,
                TaskExecutor = new ParallelTaskExecutor(),
                Reinsertion = new ElitistReinsertion(),
            };

            ga.GenerationRan += (sender, e) =>
            {
                var bestFitness = ga.BestChromosome.Fitness;
                Console.WriteLine($"Generation {ga.GenerationsNumber}: Best Fitness = {bestFitness}");
            };

            Console.WriteLine("Starting genetic algorithm with optimized operators...");
            ga.Start();

            Console.WriteLine($"Best solution found has {ga.BestChromosome.Fitness} fitness.");
            var bestChromosome = ga.BestChromosome as LlmTeamChromosome;
            if (bestChromosome != null)
            {
                Console.WriteLine($"Best genome: {string.Join(", ", bestChromosome.GetDna())}");
            }
        }
        else
        {
            Console.WriteLine("Failed to find optimized operators using AutoConfig.");
        }
    }
}

// Custom AutoConfigChromosome that includes only compatible operators
public sealed class CustomAutoConfigChromosome : ChromosomeBase
{
    private static readonly IRandomization s_randomization = RandomizationProvider.Current;

    private static readonly IList<string> s_availableSelections = new List<string>
    {
        "EliteSelection",
        "TournamentSelection",
        "RouletteWheelSelection"
    };

    private static readonly IList<string> s_availableCrossovers = new List<string>
    {
        "UniformCrossover",
        "OnePointCrossover",
        "TwoPointCrossover"
    };

    private static readonly IList<string> s_availableMutations = new List<string>
    {
        "UniformMutation",
        "ReverseSequenceMutation",
        "TworsMutation"
    };

    public CustomAutoConfigChromosome() : base(3)
    {
        CreateGenes();
    }

    public ISelection Selection
    {
        get
        {
            return GetGene(0).Value as ISelection ?? throw new InvalidOperationException("Selection gene is not of type ISelection.");
        }
    }

    public ICrossover Crossover
    {
        get
        {
            return GetGene(1).Value as ICrossover ?? throw new InvalidOperationException("Crossover gene is not of type ICrossover.");
        }
    }

    public IMutation Mutation
    {
        get
        {
            return GetGene(2).Value as IMutation ?? throw new InvalidOperationException("Mutation gene is not of type IMutation.");
        }
    }

    public override IChromosome CreateNew()
    {
        return new CustomAutoConfigChromosome();
    }

    public override Gene GenerateGene(int geneIndex)
    {
        switch (geneIndex)
        {
            // Selection.
            case 0:
                return CreateRandomGene<ISelection>(s_availableSelections);

            // Crossover.
            case 1:
                return CreateRandomGene<ICrossover>(s_availableCrossovers);

            // Mutation.
            case 2:
                return CreateRandomGene<IMutation>(s_availableMutations);

            default:
                throw new InvalidOperationException("Invalid AutoConfigChromosome gene index.");
        }
    }

    private static Gene CreateRandomGene<TGeneValue>(IList<string> available)
    {
        return new Gene(TypeHelper.CreateInstanceByName<TGeneValue>(available[s_randomization.GetInt(0, available.Count)]));
    }
}
