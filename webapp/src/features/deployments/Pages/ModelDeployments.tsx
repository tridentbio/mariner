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
import { useEffect, useRef, useState } from 'react';
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

  //? Avoids React to undetected recent hook values when `handleClickDelete` callback is called
  const deploymentsRef = useRef(deployments);

  useEffect(() => {
    deploymentsRef.current = deployments;
  }, [deployments]);

  const dispatch = useAppDispatch();
  const handleClickDelete = (id: number) => {
    const deployment = deploymentsRef.current.find((item) => item.id === id);
    if (!deployment) return;
    dispatch(setCurrentDeployment(deployment));
    setShowDeleteConfirmation(true);
  };
  const confirmDelete = async () => {
    if (currentDeployment) {
      await deleteDeployment({ deploymentId: currentDeployment.id });
      dispatch(cleanCurrentDeployment());
      setShowDeleteConfirmation(false);
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
        open={showDeleteConfirmation}
        onResult={result => {
          if (result === 'confirmed') confirmDelete();
          setShowDeleteConfirmation(false)
        }}
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
        <DeploymentsTable
          toggleModal={toggleModal}
          handleClickDelete={handleClickDelete}
          fixedTab={3}
        />
      </Box>
    </>
  );
};

export { ModelDeployments };
