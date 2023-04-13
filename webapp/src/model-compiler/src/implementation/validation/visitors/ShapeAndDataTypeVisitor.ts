import {
  ComponentType,
  FeaturizersType,
  LayersType,
} from '../../../interfaces/model-editor';
import { unwrapDollar } from '../../../utils';
import { getDependenciesNames } from '../../modelSchemaQuery';
import TransversalInfo from '../TransversalInfo';
import ComponentVisitor from './ComponentVisitor';

export default class ShapeAndDataTypeVisitor extends ComponentVisitor {
  visitMolFeaturizer: ComponentVisitor['visitMolFeaturizer'] = (
    component,
    type,
    info
  ) => {
    if (!component.constructorArgs) return;
    if (
      component.constructorArgs.sym_bond_list &&
      component.constructorArgs.allow_unknown
    ) {
      info.setShape(component.name, 'x', [1, 30]);
    } else {
      info.setShape(component.name, 'x', [1, 26]);
    }
  };

  visitGlobalPooling: ComponentVisitor['visitGlobalPooling'] = (
    component,
    type,
    info
  ) => {
    const x = component.forwardArgs.x;
    if (!x) return;
    const dep = unwrapDollar(x);
    const xShape = info.getShapeSimple(dep);
    if (xShape) info.setShapeSimple(component.name, xShape);
  };

  visitLinear: ComponentVisitor['visitLinear'] = (component, type, info) => {
    if (component.constructorArgs.out_features)
      info.setShapeSimple(component.name, [
        1,
        component.constructorArgs.out_features,
      ]);
  };

  visitGCN: ComponentVisitor['visitGCN'] = (component, type, info) => {
    if (component.constructorArgs.out_channels)
      info.setShapeSimple(component.name, [
        1,
        component.constructorArgs.out_channels,
      ]);
  };

  visitConcat: ComponentVisitor['visitConcat'] = (component, type, info) => {
    // @ts-ignore
    const deps = getDependenciesNames(component);
    let dim = component.constructorArgs.dim || 0;
    const shapes = deps.map(info.getShapeSimple);
    const shape = shapes.find((some) => !!some);
    if (!shape) return;
    const totalNewDim = shapes
      .map((shape) => (shape ? shape.at(dim) || 0 : 0))
      .reduce((acc, num) => acc + num, 0);
    let newShape = [...shape];
    if (dim < 0) dim += shape.length;
    newShape[dim] = totalNewDim;
    info.setShapeSimple(component.name, newShape);
  };

  visitOneHot: ComponentVisitor['visitOneHot'] = (component, type, info) => {};

  visitInput: ComponentVisitor['visitInput'] = (comp, type, info) => {
    info.setDataTypeSimple(comp.name, comp.dataType);
    if (comp.dataType.domainKind === 'numeric') {
      info.setShapeSimple(comp.name, [1, 1]);
    }
  };

  visitOutput: ComponentVisitor['visitOutput'] = (comp, type, info) => {};

  private visitActivation = (
    component: LayersType | FeaturizersType,
    _type: ComponentType,
    info: TransversalInfo
  ) => {
    const [dep] = getDependenciesNames(component);
    if (!dep) return;
    const shape = info.getShapeSimple(dep);
    if (shape) info.setShapeSimple(component.name, shape);
  };

  // @ts-ignore
  visitRelu: ComponentVisitor['visitRelu'] = this.visitActivation;

  // @ts-ignore
  visitSigmoid: ComponentVisitor['visitSigmoid'] = this.visitActivation;
}
