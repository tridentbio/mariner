import Suggestion from '../../Suggestion';
import ComponentVisitor from './ComponentVisitor';

export default class GlobalPoolingValidatorVisitor extends ComponentVisitor {
  visitGlobalPooling: ComponentVisitor['visitGlobalPooling'] = ({
    info,
    component,
    backward,
  }) => {
    if (backward) return;

    if (!component.forwardArgs.batch) {
      info.addSuggestion(
        new Suggestion('WARNING', 'Batch argument is not filled', [], {
          nodeId: component.name,
        })
      );
    }
  };
}
