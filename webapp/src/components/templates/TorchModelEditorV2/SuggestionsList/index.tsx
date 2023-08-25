import { AutoFixHighOutlined, ErrorSharp } from '@mui/icons-material';
import { MenuItem, Box, Alert, IconButton } from '@mui/material';
import useTorchModelEditor from 'hooks/useTorchModelEditor';
import {
  locateContext,
  SchemaContextTypeGuard,
} from 'model-compiler/src/implementation/SchemaContext';
import Suggestion from 'model-compiler/src/implementation/Suggestion';

interface SuggestionsListProps {
  suggestions: Suggestion[];
}

const ZOOM = 0.5;
const SuggestionsList = (props: SuggestionsListProps) => {
  const { setCenter, getNode, getEdge, applySuggestions, schema, expandNodes } =
    useTorchModelEditor();
  const handleClickLocate = (suggestion: Suggestion) => {
    if (SchemaContextTypeGuard.isNodeSchema(suggestion.context)) {
      expandNodes([suggestion.context.nodeId]);
    }
    const position = locateContext(suggestion.context, {
      getNode,
      getEdge,
    });
    if (!position) return;
    setCenter(position.x, position.y, {
      zoom: ZOOM,
      duration: 1000,
    });
  };
  const colorSeverity = {
    ERROR: 'error',
    WARNING: 'warning',
    IMPROV: 'info',
  } as const;
  const fixable = props.suggestions.filter(
    (suggestion) => suggestion.commands.length > 0
  );
  const handleFixAll = () => {
    schema && applySuggestions({ suggestions: props.suggestions, schema });
  };
  return (
    <Box>
      {(fixable.length > 1
        ? [
            <MenuItem key={'fix all'} onClick={handleFixAll}>
              Fix all
            </MenuItem>,
          ]
        : []
      ).concat(
        props.suggestions.map((suggestion, index) => {
          return (
            <MenuItem
              onClick={() => handleClickLocate(suggestion)}
              key={`sugg-${index}`}
            >
              <Alert
                color={colorSeverity[suggestion.severity]}
                icon={<ErrorSharp />}
              >
                {suggestion.message}
              </Alert>
              <IconButton
                onClick={(event) => {
                  event.stopPropagation();
                  schema &&
                    applySuggestions({ suggestions: [suggestion], schema });
                }}
              >
                <AutoFixHighOutlined />
              </IconButton>
            </MenuItem>
          );
        })
      )}
    </Box>
  );
};

export default SuggestionsList;
