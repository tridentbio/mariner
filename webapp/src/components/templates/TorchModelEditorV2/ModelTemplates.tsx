import {
  ModelTemplate,
  useGetModelTemplatesQuery,
} from '@app/rtk/generated/models';
import { CardActionArea, List, ListItem } from '@mui/material';
import Card, { CardProps } from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { EditorDragStartParams } from '.';

interface TemplateItemProps extends CardProps {
  template: ModelTemplate;
}

const TemplateItem = ({ template, ...rest }: TemplateItemProps) => {
  return (
    <Card sx={{ width: '100%' }} {...rest}>
      <CardActionArea>
        <CardMedia
          component="img"
          height="140"
          image=""
          alt="model-preview"
        />
        <CardContent>
          <Typography gutterBottom variant="h6" component="div">
            {template.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {template.description}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
};

interface ModelTemplatesProps {
  onDragStart?: (element: EditorDragStartParams<ModelTemplate>) => void;
}

export const ModelTemplates = ({ onDragStart }: ModelTemplatesProps) => {
  const { data: templates } = useGetModelTemplatesQuery();

  return (
    <>
      <List sx={{ p: 2, pl: 0 }}>
        {templates?.map((template, index) => (
          <ListItem sx={{ p: 0, mb: 2 }} key={index}>
            <TemplateItem
              template={template}
              draggable
              onDragStart={(event) =>
                onDragStart &&
                onDragStart({ data: template, event, type: 'Template' })
              }
            />
          </ListItem>
        ))}
      </List>
    </>
  );
};
