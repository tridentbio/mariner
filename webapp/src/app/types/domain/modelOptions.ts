export type ConstructorArgValue = Exclude<
  EPythonClasses,
  | EPythonClasses.TORCH_TENSOR_OPTIONAL
  | EPythonClasses.TORCH_TENSOR_REQUIRED
  | EPythonClasses.TORCH_GEOMETRIC_DATA_REQUIRED
>;

export interface ConstructorArgsSummary {
  aggr?: ConstructorArgValue;
  in_features?: ConstructorArgValue;
  out_features?: ConstructorArgValue;
  bias?: ConstructorArgValue;
  inplace?: ConstructorArgValue;
  in_channels?: ConstructorArgValue;
  out_channels?: ConstructorArgValue;
  improved?: ConstructorArgValue;
  cached?: ConstructorArgValue;
  add_self_loops?: ConstructorArgValue;
  normalize?: ConstructorArgValue;
  allow_unknown?: ConstructorArgValue;
  sym_bond_list?: ConstructorArgValue;
  per_atom_fragmentation?: ConstructorArgValue;
}

type ForwardArgValue =
  | Exclude<EPythonClasses, EPythonClasses.TORCH_GEOMETRIC_DATA_REQUIRED>
  | ETypings;

export interface ForwardArgsSummary {
  x1?: ForwardArgValue;
  x?: ForwardArgValue;
  batch?: ForwardArgValue;
  size?: ForwardArgValue;
  x2?: ForwardArgValue;
  input?: ForwardArgValue;
  edge_index?: ForwardArgValue;
  edge_weight?: ForwardArgValue;
  mol?: ForwardArgValue;
}

export interface ModelOptionsComponent {
  type: EClassPaths;
  constructorArgsSummary: ConstructorArgsSummary;
  forwardArgsSummary: ForwardArgsSummary;
}

export interface ModelOptions {
  docsLink: string | null;
  docs: string;
  outputType:
    | EPythonClasses.TORCH_TENSOR_REQUIRED
    | EPythonClasses.TORCH_GEOMETRIC_DATA_REQUIRED;
  classPath: EClassPaths;
  type: EModelOptionsTypes;
  component: ModelOptionsComponent;
}

export enum EModelOptionsTypes {
  LAYER = 'layer',
  FEATURIZER = 'featurizer',
}
export enum EClassPaths {
  ONE_HOT = 'model_builder.layers.OneHot',
  GLOBAL_POOLING = 'model_builder.layers.GlobalPooling',
  CONCAT = 'model_builder.layers.Concat',
  LINEAR = 'torch.nn.Linear',
  SIGMOID = 'torch.nn.Sigmoid',
  RELU = 'torch.nn.ReLU',
  GCN_CONV = 'torch_geometric.nn.GCNConv',
  MOLECULE_FEATURIZER = 'model_builder.featurizers.MoleculeFeaturizer',
}

export enum EPythonClasses {
  INT_REQUIRED = "<class 'int'>",
  INT_OPTIONAL = "<class 'int'>?",
  STR_REQUIRED = "<class 'str'>",
  STR_OPTIONAL = "<class 'str'>?",
  BOOL_REQUIRED = "<class 'bool'>",
  BOOL_OPTIONAL = "<class 'bool'>?",
  TORCH_TENSOR_REQUIRED = "<class 'torch.Tensor'>",
  TORCH_TENSOR_OPTIONAL = "<class 'torch.Tensor'>?",
  TORCH_GEOMETRIC_DATA_REQUIRED = "<class 'torch_geometric.data.data.Data'>",
}
enum ETypings {
  UNION_LIST_STR_LIST_INT = 'typing.Union[list[str], list[int]]',
  UNION_MOL_STR = 'typing.Union[rdkit.Chem.rdchem.Mol, str]',
  UNION_TENSOR_SPARSE_TENSOR = 'typing.Union[torch.Tensor, torch_sparse.tensor.SparseTensor]',
  TORCH_TENSOR_OPTIONAL_REQ = 'typing.Optional[torch.Tensor]',
  TORCH_TENSOR_OPTIONAL = 'typing.Optional[torch.Tensor]?',
  INT_OPTIONAL = 'typing.Optional[int]??',
}
