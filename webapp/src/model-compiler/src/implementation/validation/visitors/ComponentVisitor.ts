import {
  AddPooling,
  ComponentType,
  Concat,
  DNAFeaturizer,
  Embedding,
  GcnConv,
  GlobalPooling,
  Input,
  IntegerFeaturizer,
  Linear,
  MolFeaturizer,
  OneHot,
  Output,
  ProteinFeaturizer,
  Relu,
  RNAFeaturizer,
  Sigmoid,
  TransformerEncoderLayer,
} from '../../../interfaces/model-editor';
import TransversalInfo from '../TransversalInfo';

type VisitorInput<T extends {}> = {
  component: T;
  type: ComponentType;
  info: TransversalInfo;
  backward?: boolean;
};

abstract class ComponentVisitor {
  visitLinear = (_input: VisitorInput<Linear>) => {};

  visitRelu = (_input: VisitorInput<Relu>) => {};

  visitSigmoid = (_input: VisitorInput<Sigmoid>) => {};

  visitGCN = (_input: VisitorInput<GcnConv>) => {};

  visitEmbedding = (_input: VisitorInput<Embedding>) => {};

  visitTransformerEncoderLayer = (
    _input: VisitorInput<TransformerEncoderLayer>
  ) => {};

  visitConcat = (_input: VisitorInput<Concat>) => {};

  visitOneHot = (_input: VisitorInput<OneHot>) => {};

  visitGlobalPooling = (_input: VisitorInput<GlobalPooling>) => {};
  visitAddPooling = (_input: VisitorInput<AddPooling>) => {};
  visitMolFeaturizer = (_input: VisitorInput<MolFeaturizer>) => {};
  visitDNAFeaturizer = (_input: VisitorInput<DNAFeaturizer>) => {};
  visitRNAFeaturizer = (_input: VisitorInput<RNAFeaturizer>) => {};
  visitProteinFeaturizer = (_input: VisitorInput<ProteinFeaturizer>) => {};
  visitIntegerFeaturizer = (_input: VisitorInput<IntegerFeaturizer>) => {};
  visitInput = (_input: VisitorInput<Input>) => {};
  visitOutput = (_input: VisitorInput<Output>) => {};
}

export default ComponentVisitor;
