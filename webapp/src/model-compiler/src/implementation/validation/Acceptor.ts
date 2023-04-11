import { ComponentType, NodeType } from '../../interfaces/model-editor';
import TransversalInfo from './TransversalInfo';
import ComponentVisitor from './visitors/ComponentVisitor';

class Acceptor {
  accept = (
    visitor: ComponentVisitor,
    node: NodeType,
    info: TransversalInfo,
    type: ComponentType
  ) => {
    if (node.type === 'torch.nn.ReLU') {
      visitor.visitRelu(node, type, info);
    } else if (node.type === 'torch.nn.Linear') {
      visitor.visitLinear(node, type, info);
    } else if (node.type === 'torch.nn.Sigmoid') {
      visitor.visitSigmoid(node, type, info);
    } else if (node.type === 'torch_geometric.nn.GCNConv') {
      visitor.visitGCN(node, type, info);
    } else if (node.type === 'torch.nn.Embedding') {
      visitor.visitEmbedding(node, type, info);
    } else if (node.type === 'torch.nn.TransformerEncoderLayer') {
      visitor.visitTransformerEncoderLayer(node, type, info);
    } else if (node.type === 'model_builder.layers.OneHot') {
      visitor.visitOneHot(node, type, info);
    } else if (node.type === 'model_builder.layers.Concat') {
      visitor.visitConcat(node, type, info);
    } else if (node.type === 'model_builder.layers.GlobalPooling') {
      visitor.visitGlobalPooling(node, type, info);
    } else if (node.type === 'model_builder.layers.AddPooling') {
      visitor.visitAddPooling(node, type, info);
    } else if (node.type === 'model_builder.featurizers.MoleculeFeaturizer') {
      visitor.visitMolFeaturizer(node, type, info);
    } else if (
      node.type === 'model_builder.featurizers.DNASequenceFeaturizer'
    ) {
      visitor.visitDNAFeaturizer(node, type, info);
    } else if (
      node.type === 'model_builder.featurizers.RNASequenceFeaturizer'
    ) {
      visitor.visitRNAFeaturizer(node, type, info);
    } else if (
      node.type === 'model_builder.featurizers.ProteinSequenceFeaturizer'
    ) {
      visitor.visitProteinFeaturizer(node, type, info);
    } else if (node.type === 'model_builder.featurizers.IntegerFeaturizer') {
      visitor.visitIntegerFeaturizer(node, type, info);
    } else if (node.type === 'input') {
      visitor.visitInput(node, type, info);
    } else if (node.type === 'output') {
      visitor.visitOutput(node, type, info);
    } else {
      //@ts-ignore
      throw new Error(`Acceptor of ${node.type} not implemented`);
    }
  };
}

export default Acceptor;
