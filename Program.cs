using GeneticSharp;
using GeneticSharp.Extensions;
using LanguageExt;
using MathNet.Numerics.Distributions;

/// <summary>
/// Contains all constant values used throughout the application
/// </summary>
public static class Constants
{
    /// <summary>
    /// Constants related to LLM configuration and behavior
    /// </summary>
    public static class Llm
    {
        public static class Ability
        {
            public const double Mean = 0.7;
            public const double StdDev = 0.3;
            public const float Min = 0.0f;
            public const float Max = 1.0f;
        }

        public static class ContextLength
        {
            public const int DefaultLength = 8192;  // Renamed from BaseLength
            public const double LogStdDev = 2.0;
            public const int Min = 4096;
            public const int Max = 262144;
        }

        public static class Verbosity
        {
            public const double Min = 0.5;
            public const double Max = 1.5;
            public const double Default = 1.0;
        }

        public static class CostPerToken
        {
            public const double DefaultCost = 0.0000015;
            public const double StdDevMultiplier = 1.5;
            public const double Min = 0.0000006;
            public const double Max = 0.00003;
        }

        public static class Performance
        {
            public const double CompetencyNoise = 0.2;
            public const double PlanningNoise = 0.1;
            public const double SuccessThreshold = 0.5;
            public const double DefaultSuccessRate = 0.7;
        }
    }

    /// <summary>
    /// Constants related to task generation and processing
    /// </summary>
    public static class Task
    {
        public static class Generation
        {
            public const double DifficultyMean = 0.8;
            public const double DifficultyStdDev = 0.2;
            public const float DifficultyMin = 0.0f;
            public const float DifficultyMax = 1.0f;
            public const float DifficultyIncrement = 0.1f;
            public const float AverageDifficultyDivisor = 3.0f;
        }

        public static class TokenEstimates
        {
            public const double Min = 1000;
            public const double Max = 200000;
            public const double RootTaskMean = 24000;
            public const double RootTaskStdDev = 15000;
            public const double SubtaskDivisor = 10.0;
        }

        public static class Structure
        {
            public const double MinSubtasks = 1;
            public const double MaxSubtasks = 15;
            public const int DefaultSubtasks = 5;
        }
    }

    /// <summary>
    /// Constants related to genetic algorithm configuration
    /// </summary>
    public static class GeneticAlgorithm
    {
        public static class Standard
        {
            public const int PopulationMin = 100;
            public const int PopulationMax = 200;
            public const float MutationProbability = 0.2f;
            public const float CrossoverProbability = 0.8f;
            public const int GenomeLength = 6;
            public const int DefaultMaxGenerations = 10;
            public const int DefaultPopulationSize = 150;
        }

        public static class AutoConfig
        {
            public const int PopulationMin = 20;
            public const int PopulationMax = 40;
            public const float MutationProbability = 0.3f;
            public const float CrossoverProbability = 0.7f;
            public const int GeneCount = 3;
            public const int DefaultGenerations = 5;
        }

        public static class Evaluation
        {
            public const int TaskCount = 10;
            public const int TeamsPerTask = 100;
            public const int LlmTeamsPerTask = 100;
        }
    }

    /// <summary>
    /// Constants related to team configuration and formatting
    /// </summary>
    public static class Team
    {
        public const double MajorityThreshold = 0.5;

        public static class Analysis
        {
            public const int DefaultIndentSpaces = 2;
            public const int BaseIndentSpaces = 4;
        }

        public static class Format
        {
            public const string DefaultIndent = "  ";
            public const string IndentStep = "    ";
            public const int DefaultIndentLength = 2;
        }
    }

    /// <summary>
    /// Mathematical constants used throughout the application
    /// </summary>
    public static class Math
    {
        public const float Half = 0.5f;
        public const float Third = 0.333333f;
    }

    /// <summary>
    /// Program-wide default settings
    /// </summary>
    public static class Defaults
    {
        public static class Program
        {
            public const int MaxGenerations = 10;
            public const bool UseAutoConfig = false;
            public const string DefaultIndent = "  ";
        }

        public static class Validation
        {
            public const double MinProbability = 0.0;
            public const double MaxProbability = 1.0;
            public const int MinCount = 1;
        }
    }
}
public class Llm
{
    public int ContextLength { get; }
    public float PlanningAbility { get; }
    public float Verbosity { get; }
    public float Competency { get; }
    public float CostPerToken { get; }

    private static readonly Normal AbilityDistribution = new Normal(
        Constants.Llm.Ability.Mean,
        Constants.Llm.Ability.StdDev
    );

    private static readonly LogNormal ContextLengthDistribution = new LogNormal(
        Constants.Llm.ContextLength.DefaultLength,
        Constants.Llm.ContextLength.LogStdDev
    );

    private static readonly ContinuousUniform VerbosityDistribution = new ContinuousUniform(
        Constants.Llm.Verbosity.Min,
        Constants.Llm.Verbosity.Max
    );

    private static readonly LogNormal CostPerTokenDistribution = new LogNormal(
        Constants.Llm.CostPerToken.DefaultCost,
        Constants.Llm.CostPerToken.StdDevMultiplier
    );

    public Llm()
    {
        ContextLength = (int)
            Math.Clamp(
                Math.Pow(2, Math.Round(ContextLengthDistribution.Sample())),
                Constants.Llm.ContextLength.Min,
                Constants.Llm.ContextLength.Max
            );
        PlanningAbility = (float)Math.Clamp(AbilityDistribution.Sample(), 0, 1);
        Competency = (float)Math.Clamp(AbilityDistribution.Sample(), 0, 1);
        Verbosity = (float)VerbosityDistribution.Sample();
        CostPerToken = (float)
            Math.Clamp(
                CostPerTokenDistribution.Sample(),
                Constants.Llm.CostPerToken.Min,
                Constants.Llm.CostPerToken.Max
            );
    }

    public bool SolveTask(Task task)
    {
        float scaledCompetency = Competency * task.RootTask.CompetencyRequired;
        float scaledPlanningAbility = PlanningAbility * task.RootTask.PlanningRequired;
        float avgCompetency = (scaledCompetency + scaledPlanningAbility) / 2f;

        int scaledTokenEstimate = (int)(task.RootTask.TokenEstimate * Verbosity);
        float adjustedCompetency =
            scaledTokenEstimate > ContextLength ? avgCompetency / 2f : avgCompetency;

        return new Normal(adjustedCompetency, Constants.Llm.Performance.CompetencyNoise)
            .Sample()
            > Constants.Llm.Performance.SuccessThreshold;
    }

    public List<SubTask> BreakDownTask(Task task)
    {
        float threshold =
            1f - (task.RootTask.ReasoningRequired + task.RootTask.PlanningRequired) / 2f;
        bool success =
            new Normal(PlanningAbility, Constants.Llm.Performance.PlanningNoise).Sample() > threshold;

        if (success)
        {
            return task.Subtasks;
        }
        else
        {
            return task
                .Subtasks.Select(subtask => new SubTask
                {
                    ReasoningRequired = Math.Min(
                        subtask.ReasoningRequired + Constants.Task.Generation.DifficultyIncrement,
                        Constants.Task.Generation.DifficultyMax
                    ),
                    PlanningRequired = Math.Min(
                        subtask.PlanningRequired + Constants.Task.Generation.DifficultyIncrement,
                        Constants.Task.Generation.DifficultyMax
                    ),
                    CompetencyRequired = Math.Min(
                        subtask.CompetencyRequired + Constants.Task.Generation.DifficultyIncrement,
                        Constants.Task.Generation.DifficultyMax
                    ),
                    TokenEstimate = subtask.TokenEstimate,
                })
                .ToList();
        }
    }
}

public enum LLMGene
{
    Single = 0,
    Vertical = 1,
    Horizontal = 2,
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
        return Followers
            .Zip(chunks)
            .All(pair =>
            {
                var (follower, subtaskChunk) = pair;
                return subtaskChunk.All(subtask => follower.SolveTask(new Task()));
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
        return results.Count(r => r) > results.Count * Constants.Team.MajorityThreshold;
    }

    public override List<SubTask> BreakDownTask(Task task) => new List<SubTask>();
}

public class LlmTeamChromosome : ChromosomeBase
{
    public int Age { get; set; }
    public double FitnessValue { get; set; }

    public LlmTeamChromosome()
        : base(Constants.GeneticAlgorithm.AutoConfig.GeneCount)
    {
        CreateGenes();
    }

    public override Gene GenerateGene(int geneIndex)
    {
        return new Gene(
            RandomizationProvider.Current.GetInt(0, Enum.GetValues(typeof(LLMGene)).Length)
        );
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

        var sortedLlms = llms.OrderByDescending(llm => (llm.Competency + llm.PlanningAbility) / 2)
            .ToList();
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
                        teamStack.Push(
                            new VerticalLlmTeam(new SingleLlmTeam(llmIter.Current), followers)
                        );
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
    private const int TaskCount = Constants.GeneticAlgorithm.Evaluation.TaskCount;
    private const int LlmTeamsPerTask = Constants.GeneticAlgorithm.Evaluation.TeamsPerTask;

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
                    None: () => { /* Invalid team, do nothing */
                    }
                );
            }
        }

        genome.FitnessValue = totalFitness;
        return totalFitness;
    }
}

public class SubTask
{
    private static readonly Normal DifficultyDistribution = new Normal(
        Constants.Task.Generation.DifficultyMean,
        Constants.Task.Generation.DifficultyStdDev
    );
    private static readonly ContinuousUniform TokenEstimateDistribution = new ContinuousUniform(
        Constants.Task.TokenEstimates.Min,
        Constants.Task.TokenEstimates.Max
    );

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
    private static readonly Normal RootTaskTokenEstimateDistribution = new Normal(
        Constants.Task.TokenEstimates.RootTaskMean,
        Constants.Task.TokenEstimates.RootTaskStdDev
    );
    private static readonly ContinuousUniform SubtaskCountDistribution = new ContinuousUniform(
        Constants.Task.Structure.MinSubtasks,
        Constants.Task.Structure.MaxSubtasks
    );

    public SubTask RootTask { get; set; }
    public List<SubTask> Subtasks { get; set; }
    public float EconomicValue { get; set; }
    public List<List<Llm>> Llms { get; set; }

    public Task()
    {
        RootTask = new SubTask
        {
            TokenEstimate = (int)
                Math.Clamp(
                    RootTaskTokenEstimateDistribution.Sample(),
                    Constants.Task.TokenEstimates.Min,
                    Constants.Task.TokenEstimates.Max
                ),
        };

        int numSubtasks = (int)SubtaskCountDistribution.Sample();
        float avgSubtaskTokenEstimate = RootTask.TokenEstimate / (float)numSubtasks;

        Subtasks = Enumerable
            .Range(0, numSubtasks)
            .Select(_ => new SubTask
            {
                TokenEstimate = (int)
                    Math.Clamp(
                        new Normal(
                            avgSubtaskTokenEstimate,
                            Constants.Task.TokenEstimates.RootTaskStdDev
                        ).Sample(),
                        (int)Constants.Task.TokenEstimates.Min / numSubtasks,
                        (int)Constants.Task.TokenEstimates.Max / numSubtasks
                    ),
            })
            .ToList();

        float avgDifficulty =
            (RootTask.ReasoningRequired + RootTask.PlanningRequired + RootTask.CompetencyRequired)
            / 3f;
        EconomicValue = (1f - avgDifficulty) * RootTask.TokenEstimate;

        Llms = Enumerable
            .Range(0, Constants.GeneticAlgorithm.Evaluation.TeamsPerTask)
            .Select(_ =>
                Enumerable
                    .Range(0, Constants.GeneticAlgorithm.AutoConfig.GeneCount)
                    .Select(_ => new Llm())
                    .ToList()
            )
            .ToList();
    }

    public static List<Task> GenerateTasks(int amount) =>
        Enumerable.Range(0, amount).Select(_ => new Task()).ToList();
}

class Program
{
    static void PrintTeamStructure(LlmTeam team, string indent = "")
    {
        string baseIndent = "";
        string indentStep = "  ";

        switch (team)
        {
            case SingleLlmTeam singleTeam:
                Console.WriteLine(
                    $"{indent}Single LLM (Competency: {singleTeam.Llm.Competency:F2}, Planning Ability: {singleTeam.Llm.PlanningAbility:F2})"
                );
                break;
            case VerticalLlmTeam verticalTeam:
                Console.WriteLine($"{indent}Vertical LLM Team:");
                Console.WriteLine($"{indent}{baseIndent}Leader:");
                PrintTeamStructure(verticalTeam.Leader, indent + indentStep);
                Console.WriteLine($"{indent}{baseIndent}Followers:");
                foreach (var follower in verticalTeam.Followers)
                {
                    PrintTeamStructure(follower, indent + indentStep);
                }
                break;
            case HorizontalLlmTeam horizontalTeam:
                Console.WriteLine($"{indent}Horizontal LLM Team:");
                foreach (var member in horizontalTeam.Members)
                {
                    PrintTeamStructure(member, indent + baseIndent);
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
            if (
                args[i] == "-m"
                && i + 1 < args.Length
                && int.TryParse(args[i + 1], out int parsedValue)
            )
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
            if (llmChromosome == null)
                continue;

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
        var population = new Population(
            Constants.GeneticAlgorithm.Standard.PopulationMin,
            Constants.GeneticAlgorithm.Standard.PopulationMax,
            chromosome
        );

        var ga = new GeneticAlgorithm(population, fitness, selection, crossover, mutation)
        {
            Termination = new GenerationNumberTermination(maxGenerations),
            MutationProbability = Constants.GeneticAlgorithm.Standard.MutationProbability,
            CrossoverProbability = Constants.GeneticAlgorithm.Standard.CrossoverProbability,
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
        var targetFitness = new LlmTeamFitness();
        var targetChromosome = new LlmTeamChromosome();
        var autoConfigFitness = new AutoConfigFitness(targetFitness, targetChromosome);

        var autoConfigPopulation = new Population(
            Constants.GeneticAlgorithm.AutoConfig.PopulationMin,
            Constants.GeneticAlgorithm.AutoConfig.PopulationMax,
            new CustomAutoConfigChromosome()
        );

        var autoConfigGa = new GeneticAlgorithm(
            autoConfigPopulation,
            autoConfigFitness,
            new EliteSelection(),
            new UniformCrossover(),
            new UniformMutation()
        )
        {
            Termination = new GenerationNumberTermination(maxGenerations / 2),
            MutationProbability = Constants.GeneticAlgorithm.AutoConfig.MutationProbability,
            CrossoverProbability = Constants.GeneticAlgorithm.AutoConfig.CrossoverProbability,
            TaskExecutor = new ParallelTaskExecutor(),
        };

        autoConfigGa.GenerationRan += (sender, e) =>
        {
            var bestAutoChromosome = autoConfigGa.BestChromosome as CustomAutoConfigChromosome;
            Console.WriteLine(
                $"Meta Generation {autoConfigGa.GenerationsNumber}: Best Meta Fitness = {bestAutoChromosome?.Fitness}"
            );
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
                Console.WriteLine(
                    $"Generation {ga.GenerationsNumber}: Best Fitness = {bestFitness}"
                );
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
        "RouletteWheelSelection",
    };

    private static readonly IList<string> s_availableCrossovers = new List<string>
    {
        "UniformCrossover",
        "OnePointCrossover",
        "TwoPointCrossover",
    };

    private static readonly IList<string> s_availableMutations = new List<string>
    {
        "UniformMutation",
        "ReverseSequenceMutation",
        "TworsMutation",
    };

    public CustomAutoConfigChromosome()
        : base(Constants.GeneticAlgorithm.AutoConfig.GeneCount)
    {
        CreateGenes();
    }

    public ISelection Selection
    {
        get
        {
            return GetGene(0).Value as ISelection
                ?? throw new InvalidOperationException("Selection gene is not of type ISelection.");
        }
    }

    public ICrossover Crossover
    {
        get
        {
            return GetGene(1).Value as ICrossover
                ?? throw new InvalidOperationException("Crossover gene is not of type ICrossover.");
        }
    }

    public IMutation Mutation
    {
        get
        {
            return GetGene(2).Value as IMutation
                ?? throw new InvalidOperationException("Mutation gene is not of type IMutation.");
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
        return new Gene(
            TypeHelper.CreateInstanceByName<TGeneValue>(
                available[s_randomization.GetInt(0, available.Count)]
            )
        );
    }
}
