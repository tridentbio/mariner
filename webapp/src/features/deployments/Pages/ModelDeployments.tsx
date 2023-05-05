import { Model } from 'app/types/domain/models';
import { Box, Button } from '@mui/material';
import DeploymentsTable from '../Components/DeploymentsTable';
import Modal from 'components/templates/Modal';
import { useToggle } from 'hooks/useToogle';
import { DeploymentForm } from '../Components/DeploymentForm';
import { useAppDispatch, useAppSelector } from 'app/hooks';
import {
  cleanCurrentDeployment,
  setCurrentDeployment,
} from '../deploymentsSlice';
import ConfirmationDialog from 'components/templates/ConfirmationDialog';
import { useState } from 'react';
import * as deploymentsApi from 'app/rtk/generated/deployments';
interface ModelDeploymentsProps {
  model: Model;
}

const ModelDeployments = ({ model }: ModelDeploymentsProps) => {
  const [openModal, toggleModal] = useToggle();
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false);
  const [deleteDeployment] = deploymentsApi.useDeleteDeploymentMutation();
  const [currentDeployment, deployments] = useAppSelector((state) => [
    state.deployments.current,
    state.deployments.deployments,
  ]);
  const dispatch = useAppDispatch();
  const handleClickDelete = (id: number) => {
    const deployment = deployments.find((item) => item.id === id);
    if (!deployment) return;
    dispatch(setCurrentDeployment(deployment));
    setShowDeleteConfirmation(true);
  };
  const confirmDelete = async () => {
    if (currentDeployment) {
      await deleteDeployment({ deploymentId: currentDeployment.id });
      dispatch(cleanCurrentDeployment());
    }
  };

  return (
    <>
      <Modal
        open={openModal}
        closeOnlyOnClickX
        onClose={() => {
          toggleModal();
          if (currentDeployment) dispatch(cleanCurrentDeployment());
        }}
        title={currentDeployment ? 'EDIT DEPLOYMENT' : 'CREATE NEW DEPLOYMENT'}
      >
        <DeploymentForm {...{ model, toggleModal }} />
      </Modal>
      <ConfirmationDialog
        title="Confirm delete deployment"
        text={'Are you sure to delete this deployment? '}
        alertText="Be aware that won't be possible to recover it."
        handleClickConfirm={confirmDelete}
        open={showDeleteConfirmation}
        setOpen={setShowDeleteConfirmation}
      />
      <Box sx={{ ml: 'auto', mr: 'auto' }}>
        <Button
          sx={{
            float: 'left',
            mb: '5px',
          }}
          variant="contained"
          onClick={() => toggleModal()}
        >
          Deploy Model
        </Button>
        <DeploymentsTable {...{ toggleModal, handleClickDelete }} />
      </Box>
    </>
  );
};

export { ModelDeployments };
