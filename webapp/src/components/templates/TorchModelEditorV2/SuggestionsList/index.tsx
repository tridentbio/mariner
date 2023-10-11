import {
  AutoFixHighOutlined,
  ErrorSharp,
  Visibility,
  Error,
} from '@mui/icons-material';
import {
  ListItem,
  Box,
  Alert,
  IconButton,
  ButtonGroup,
  List,
  Button,
} from '@mui/material';
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
    <List sx={{ pt: 2 }}>
      {(fixable.length > 1
        ? [
            <ListItem key={'fix all'}>
              <Button onClick={handleFixAll}>Fix all</Button>
            </ListItem>,
          ]
        : []
      ).concat(
        props.suggestions.map((suggestion, index) => {
          return (
            <ListItem key={`sugg-${index}`}>
              <Alert
                color={colorSeverity[suggestion.severity]}
                icon={<ErrorSharp />}
                sx={{
                  py: 0,
                  alignItems: 'center',
                  '.MuiAlert-message': {
                    p: 0,
                  },
                }}
              >
                <Box display="flex" alignItems="center">
                  {suggestion.message}

                  <ButtonGroup
                    variant="outlined"
                    sx={{
                      ml: 2,
                      p: 1,
                    }}
                    aria-label="outlined button group"
                  >
                    {suggestion.commands.length > 0 ? (
                      <IconButton
                        onClick={(event) => {
                          schema &&
                            applySuggestions({
                              suggestions: [suggestion],
                              schema,
                            });
                        }}
                      >
                        <AutoFixHighOutlined fontSize="small" />
                      </IconButton>
                    ) : null}
                    <IconButton
                      onClick={() => {
                        handleClickLocate(suggestion);
                      }}
                    >
                      <Visibility fontSize="small" />
                    </IconButton>
                  </ButtonGroup>
                </Box>
              </Alert>
            </ListItem>
          );
        })
      )}
    </List>
  );
};

export default SuggestionsList;
