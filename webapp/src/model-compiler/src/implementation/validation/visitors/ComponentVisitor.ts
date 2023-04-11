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

abstract class ComponentVisitor {
  visitLinear = (
    _component: Linear,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitRelu = (
    _component: Relu,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitSigmoid = (
    _component: Sigmoid,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitGCN = (
    _component: GcnConv,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitEmbedding = (
    _component: Embedding,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitTransformerEncoderLayer = (
    _component: TransformerEncoderLayer,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitConcat = (
    _component: Concat,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitOneHot = (
    _component: OneHot,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};

  visitGlobalPooling = (
    _component: GlobalPooling,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitAddPooling = (
    _component: AddPooling,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitMolFeaturizer = (
    _component: MolFeaturizer,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitDNAFeaturizer = (
    _component: DNAFeaturizer,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitRNAFeaturizer = (
    _component: RNAFeaturizer,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitProteinFeaturizer = (
    _component: ProteinFeaturizer,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitIntegerFeaturizer = (
    _component: IntegerFeaturizer,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitInput = (
    _component: Input,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
  visitOutput = (
    _component: Output,
    _type: ComponentType,
    _info: TransversalInfo
  ) => {};
}

export default ComponentVisitor;
