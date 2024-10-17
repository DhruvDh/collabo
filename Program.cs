using GeneticSharp;
using MathNet.Numerics.Distributions;
using LanguageExt;
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
    public List<LLMGene> Dna { get; private set; }
    public int Age { get; set; }
    public double FitnessValue { get; set; }

    private const int GenomeLength = 6;

    public LlmTeamChromosome() : base(GenomeLength)
    {
        Dna = new List<LLMGene>(GenomeLength);
        for (int i = 0; i < GenomeLength; i++)
        {
            Dna.Add((LLMGene)GenerateGene(i).Value);
        }
    }

    public override Gene GenerateGene(int geneIndex)
    {
        return new Gene(RandomizationProvider.Current.GetInt(0, 3));
    }
    public override IChromosome CreateNew()
    {
        return new LlmTeamChromosome();
    }

    public override IChromosome Clone()
    {
        var clone = new LlmTeamChromosome
        {
            Dna = new List<LLMGene>(Dna),
            Age = Age,
            FitnessValue = FitnessValue
        };
        return clone;
    }

    public Option<LlmTeam> ParseGenotype(List<Llm> llms)
    {
        if (Dna == null || Dna.Count == 0)
        {
            return Option<LlmTeam>.None;
        }

        var sortedLlms = llms.OrderByDescending(llm => (llm.Competency + llm.PlanningAbility) / 2).ToList();
        var teamStack = new Stack<LlmTeam>();
        var llmIter = sortedLlms.GetEnumerator();

        if (Dna[0] == LLMGene.Single)
        {
            if (llmIter.MoveNext())
            {
                return Option<LlmTeam>.Some(new SingleLlmTeam(llmIter.Current));
            }
            return Option<LlmTeam>.None;
        }

        for (int i = Dna.Count - 1; i >= 1; i--)
        {
            switch (Dna[i])
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
    static void Main(string[] args)
    {
        int maxGenerations = 10;
        if (args.Length > 1 && args[0] == "-m" && int.TryParse(args[1], out int parsedValue))
        {
            maxGenerations = parsedValue;
        }

        var selection = new TournamentSelection();
        var crossover = new CycleCrossover();
        var mutation = new UniformMutation();
        var fitness = new LlmTeamFitness();
        var chromosome = new LlmTeamChromosome();
        var population = new Population(50, 70, chromosome);

        var ga = new GeneticAlgorithm(population, fitness, selection, crossover, mutation)
        {
            Termination = new GenerationNumberTermination(maxGenerations),
            MutationProbability = 0.2f,
            CrossoverProbability = 0.8f,
            TaskExecutor = new ParallelTaskExecutor()
        };

        Console.WriteLine("Starting genetic algorithm...");
        ga.Start();

        Console.WriteLine($"Best solution found has {ga.BestChromosome.Fitness} fitness.");
        var bestChromosome = ga.BestChromosome as LlmTeamChromosome;
        if (bestChromosome != null)
        {
            Console.WriteLine($"Best genome: {string.Join(", ", bestChromosome.Dna)}");
        }
    }
}