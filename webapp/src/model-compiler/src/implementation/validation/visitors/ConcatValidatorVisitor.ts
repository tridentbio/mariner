import { arrayEquals, normalizeNegativeIndex } from '../../../utils';
import { getDependenciesNames } from '../../modelSchemaQuery';
import Suggestion from '../../Suggestion';
import ComponentVisitor from './ComponentVisitor';

class ConcatValidatorVisitor extends ComponentVisitor {
  visitConcat: ComponentVisitor['visitConcat'] = (input) => {
    if (input.backward) return this.visitConcatBackward(input);
    else return this.visitConcatForward(input);
  };

  visitConcatForward: ComponentVisitor['visitConcat'] = ({
    component,
    info,
  }) => {
    // @ts-ignore
    const deps = getDependenciesNames(component);
    const shapes = deps
      .map(info.getShapeSimple)
      .filter((shape) => !!shape) as number[][]; // filter unknown shapes
    const dim = component.constructorArgs.dim || 0;
    if (!this.allTheSameExceptOn(shapes, dim)) {
      info.addSuggestion(
        Suggestion.makeForwardArgsError(
          { nodeId: component.name },
          'Dimensions of inputs must match except on dimension dim'
        )
      );
    }
  };

  visitConcatBackward: ComponentVisitor['visitConcat'] = ({
    component,
    info,
  }) => {};

  private allTheSameExceptOn = (shapes: number[][], dim: number) => {
    if (!shapes.length) {
      return true;
    }
    let [shape] = shapes;
    dim = normalizeNegativeIndex(dim, shape.length);
    const targetShape = [...shape];
    targetShape[dim] = 0;
    return shapes
      .map((shape) => {
        let dimMasked = [...shape];
        dimMasked[dim] = 0;
        return dimMasked;
      })
      .reduce((acc, val) => {
        const itemVal = arrayEquals(targetShape, val);
        return acc && itemVal;
      }, true);
  };
}

export default ConcatValidatorVisitor;
