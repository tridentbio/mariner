import { CardActionArea, List, ListItem } from '@mui/material';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';

const TemplateItem = () => {
  return (
    <Card>
      <CardActionArea>
        <CardMedia
          component="img"
          height="140"
          image="/static/images/cards/contemplative-reptile.jpg"
          alt="model-preview"
        />
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
            Model
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Lorem ipsum dolor sit, amet consectetur adipisicing elit. Itaque,
            fugit quibusdam
          </Typography>
        </CardContent>
      </CardActionArea>
      {/*  <CardActions>
        <Button size="small" color="primary">
          Share
        </Button>
      </CardActions> */}
    </Card>
  );
};

interface ModelTemplatesProps {}

export const ModelTemplates = ({}: ModelTemplatesProps) => {
  return (
    <List sx={{ p: 2 }}>
      <ListItem>
        <TemplateItem />
      </ListItem>
      <ListItem>
        <TemplateItem />
      </ListItem>
    </List>
  );
};
