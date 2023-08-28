import { ComponentType, NodeType } from '../../interfaces/torch-model-editor';
import TransversalInfo from './TransversalInfo';
import ComponentVisitor from './visitors/ComponentVisitor';

class Acceptor {
  accept = (
    visitor: ComponentVisitor,
    visitorInput: {
      component: NodeType;
      info: TransversalInfo;
      type: ComponentType;
      backward?: boolean;
    }
  ) => {
    const { component, type, info, backward } = visitorInput;
    if (component.type === 'torch.nn.ReLU') {
      visitor.visitRelu({ component, type, info, backward });
    } else if (component.type === 'torch.nn.Linear') {
      visitor.visitLinear({
        component,
        type,
        info,
        backward,
      });
    } else if (component.type === 'torch.nn.Sigmoid') {
      visitor.visitSigmoid({ component, type, info, backward });
    } else if (component.type === 'torch_geometric.nn.GCNConv') {
      visitor.visitGCN({ component, type, info, backward });
    } else if (component.type === 'torch.nn.Embedding') {
      visitor.visitEmbedding({ component, type, info, backward });
    } else if (component.type === 'torch.nn.TransformerEncoderLayer') {
      visitor.visitTransformerEncoderLayer({ component, type, info, backward });
    } else if (component.type === 'fleet.model_builder.layers.OneHot') {
      visitor.visitOneHot({ component, type, info, backward });
    } else if (component.type === 'fleet.model_builder.layers.Concat') {
      visitor.visitConcat({ component, type, info, backward });
    } else if (component.type === 'fleet.model_builder.layers.GlobalPooling') {
      visitor.visitGlobalPooling({ component, type, info, backward });
    } else if (component.type === 'fleet.model_builder.layers.AddPooling') {
      visitor.visitAddPooling({ component, type, info, backward });
    } else if (
      component.type === 'fleet.model_builder.featurizers.MoleculeFeaturizer'
    ) {
      visitor.visitMolFeaturizer({ component, type, info, backward });
    } else if (
      component.type === 'fleet.model_builder.featurizers.DNASequenceFeaturizer'
    ) {
      visitor.visitDNAFeaturizer({ component, type, info, backward });
    } else if (
      component.type === 'fleet.model_builder.featurizers.RNASequenceFeaturizer'
    ) {
      visitor.visitRNAFeaturizer({ component, type, info, backward });
    } else if (
      component.type ===
      'fleet.model_builder.featurizers.ProteinSequenceFeaturizer'
    ) {
      visitor.visitProteinFeaturizer({ component, type, info, backward });
    } else if (
      component.type === 'fleet.model_builder.featurizers.IntegerFeaturizer'
    ) {
      visitor.visitIntegerFeaturizer({ component, type, info, backward });
    } else if (component.type === 'input') {
      visitor.visitInput({ component, type, info, backward });
    } else if (component.type === 'output') {
      visitor.visitOutput({ component, type, info, backward });
    } else {
      //@ts-ignore
      throw new Error(`Acceptor of ${component.type} not implemented`);
    }
  };
}

export default Acceptor;
