import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from '@mui/material';
import { Text } from '../../../../components/molecules/Text';

interface DatasetStatsProps {
  stats: {
    [key: string]: {
      [key: string]: number | string;
    };
  };
  replace?: (val: number | string) => number | string;
}
const DatasetStats = (props: DatasetStatsProps) => {
  const getRows = () => {
    return Object.keys(props.stats).map((k) => ({
      name: k,
      ...props.stats[k],
    }));
  };

  const getColumns = (): string[] => {
    const keys = Object.keys(props.stats);

    if (!keys.length) return ['name'];
    const statKeys = Object.keys(props.stats[keys[0]] || {});

    return ['name', ...statKeys];
  };

  if (!props.stats) {
    return <Text>No metadata found for dataset...</Text>;
  }

  return (
    <Paper sx={{ overflowX: 'auto' }}>
      <Table>
        <TableHead>
          <TableRow>
            {getColumns().map((col) => (
              <TableCell key={col}>{col}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {getRows().map(
            (row: { name: string; [key: string]: string | number }) => (
              <TableRow key={row.name}>
                {getColumns().map((col) => (
                  <TableCell key={`${row.name}-${col}`}>
                    {col in row
                      ? (props.replace && props.replace(row[col])) || row[col]
                      : ''}
                  </TableCell>
                ))}
              </TableRow>
            )
          )}
        </TableBody>
      </Table>
    </Paper>
  );
};

export default DatasetStats;
