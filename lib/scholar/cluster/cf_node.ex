defmodule Scholar.Cluster.CFNode do
  import Nx.Defn
  import Scholar.Shared
  require Nx
  
  @derive {Nx.Container, containers: [
    :threshold,
    :branching_factor,
    :is_leaf,
    :n_features,
    :subclusters, 
    :prev_leaf, 
    :next_leaf, 
    :centroids, 
    :squared_norm
  ]}
  defstruct [
    :threshold,
    :branching_factor,
    :is_leaf,
    :n_features,
    :subclusters, 
    :prev_leaf, 
    :next_leaf, 
    :centroids, 
    :squared_norm
  ]


  opts = [
    threshold: [
      type: :float,
      default: 0.5,
      doc: """
      The radius of the subcluster obtained by merging a new sample and the closest 
      subcluster should be lesser than the threshold. Otherwise a new subcluster is started. 
      Setting this value to be very low promotes splitting and vice-versa.
      """
    ],
    branching_factor: [
      type: :pos_integer,
      default: 50,
      doc: """
      Maximum number of CF subclusters in each node. If a new samples enters such that the number 
      of subclusters exceed the branching_factor then that node is split into two nodes with the 
      subclusters redistributed in each. The parent subcluster of that node is removed and two new 
      subclusters are added as parents of the 2 split nodes.
      """
    ],
    is_leaf: [
      type: :boolean,
      default: false,
      doc: "The number of clusters to form as well as the number of centroids to generate."
    ],
    n_features: [
      required: true,
      type: :pos_integer,
      doc: "The number of clusters to form as well as the number of centroids to generate."
    ],
  ]

  @opts_schema NimbleOptions.new!(opts)


  defn new(opts \\ []) do
    threshold = opts[:threshold]
    branching_factor = opts[:branching_factor]
    n_features = opts[:n_features]
    subclusters = Nx.broadcast(Nx.tensor(0), {branching_factor,})
    centroids = Nx.broadcast(Nx.tensor(0), {branching_factor + 1, n_features})
    squared_norm = Nx.broadcast(Nx.tensor(0), {branching_factor + 1,})
    %__MODULE__{
    threshold: threshold,
    branching_factor: branching_factor,
    n_features: n_features,
    subclusters:  subclusters,
    # prev_leaf: nil, 
    # next_leaf: nil, 
    centroids: centroids, 
    squared_norm: squared_norm, 
    }
  end

  defn append_subcluster(node, subcluster) do 
      
  end 



end