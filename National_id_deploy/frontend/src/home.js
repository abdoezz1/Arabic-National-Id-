// ✅ FINAL REACT FRONTEND (home.js) for ID Upload and Display
import { useState, useEffect } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import {
  CardActionArea,
  CardMedia,
  Grid,
  TableContainer,
  Table,
  TableBody,
  TableRow,
  TableCell,
  Button,
  CircularProgress,
  Box
} from "@material-ui/core";
import { DropzoneArea } from 'material-ui-dropzone';
import PermIdentityIcon from '@material-ui/icons/PermIdentity';
import image from "./download.jpg";
const axios = require("axios").default;

const ColorButton = withStyles(() => ({
  root: {
    color: '#ffffff',
    backgroundColor: '#1f2937',
    '&:hover': {
      backgroundColor: '#374151',
    },
  },
}))(Button);

const useStyles = makeStyles(() => ({
  appbar: {
    background: '#1f2937',
    boxShadow: 'none',
    color: 'white'
  },
  grow: { flexGrow: 1 },
  mainContainer: {
    backgroundImage: `url(${image})`,
    backgroundRepeat: 'no-repeat',
    backgroundSize: '100% 100%',
    backgroundAttachment: 'fixed',
    minHeight: '100vh',
    paddingTop: '1em',
  },
  media: {
    height: 400,
    backgroundSize: 'contain',
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
  },
  imageCard: {
    width: 400,
    backgroundColor: '#1f2937',
    borderRadius: '15px'
  },
  dataCard: {
    width: 400,
    backgroundColor: '#1f2937',
    borderRadius: '15px',
    padding: '1em'
  },
  detail: {
    display: 'flex',
    justifyContent: 'center',
    flexDirection: 'column',
    alignItems: 'center',
    color: '#ffffff',
  },
  tableCell: {
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#ffffff',
  },
  loader: {
    color: '#ffffff !important',
  },
  clearButton: {
    width: '100%',
    padding: '15px',
    fontWeight: 'bold',
    backgroundColor: '#1f2937',
  },
  processingBox: {
    backgroundColor: '#1f2937',
    padding: '10px 20px',
    borderRadius: '10px',
    marginTop: '10px'
  }
}));

export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
  const [isLoading, setIsloading] = useState(false);

  const sendFile = async () => {
    if (image) {
      const formData = new FormData();
      formData.append("file", selectedFile);
      try {
        const res = await axios.post("http://127.0.0.1:8000/predict", formData);
        if (res.status === 200) {
          console.log("🧾 Received from backend:", res.data);
          setData(res.data);
        }
      } catch (error) {
        console.error("Error uploading image:", error);
      } finally {
        setIsloading(false);
      }
    }
  };

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) return;
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    let isMounted = true;
    if (!preview) return;
    setIsloading(true);
    sendFile().then(() => {
      if (!isMounted) setIsloading(false);
    });
    return () => {
      isMounted = false;
    };
  }, [preview]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) return;
    setSelectedFile(files[0]);
    setImage(true);
    setData(undefined);
  };

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography variant="h6" noWrap>
            ID Extraction System
          </Typography>
          <div className={classes.grow} />
          <PermIdentityIcon fontSize="large" />
        </Toolbar>
      </AppBar>

      <Container maxWidth={false} className={classes.mainContainer}>
        <Grid container justify="center" alignItems="flex-start" spacing={4}>
          <Grid item>
            {!data ? (
              <Card className={classes.imageCard}>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom style={{ color: "white" }}>
                    Upload your ID here
                  </Typography>
                  <DropzoneArea
                    acceptedFiles={['image/*']}
                    dropzoneText={"Upload a National ID card image"}
                    onChange={onSelectFile}
                  />
                </CardContent>
              </Card>
            ) : (
              <Card className={classes.imageCard}>
                <CardActionArea>
                  <CardMedia
                    className={classes.media}
                    image={`http://localhost:8000/${data.id_image_path}`}
                  />
                </CardActionArea>
              </Card>
            )}

            {isLoading && (
              <CardContent className={classes.detail}>
                <CircularProgress className={classes.loader} />
                <Box className={classes.processingBox}>
                  <Typography variant="h6" style={{ color: '#ffffff' }}>Processing...</Typography>
                </Box>
              </CardContent>
            )}
          </Grid>

          {data && (
            <Grid item>
              <Card className={classes.dataCard}>
                <Typography variant="h6" style={{ color: 'white' }}>Extracted Information</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow><TableCell className={classes.tableCell}>National ID</TableCell><TableCell className={classes.tableCell}>{data.national_id}</TableCell></TableRow>
                      <TableRow><TableCell className={classes.tableCell}>First Name</TableCell><TableCell className={classes.tableCell}>{data.first_name}</TableCell></TableRow>
                      <TableRow><TableCell className={classes.tableCell}>Last Name</TableCell><TableCell className={classes.tableCell}>{data.last_name}</TableCell></TableRow>
                      <TableRow><TableCell className={classes.tableCell}>Address 1</TableCell><TableCell className={classes.tableCell}>{data.address1}</TableCell></TableRow>
                      <TableRow><TableCell className={classes.tableCell}>Address 2</TableCell><TableCell className={classes.tableCell}>{data.address2}</TableCell></TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Card>
            </Grid>
          )}
        </Grid>

        {data && (
          <Grid container justify="center" style={{ marginTop: "1em" }}>
            <Grid item xs={4}>
              <ColorButton onClick={clearData} className={classes.clearButton}>
                Clear
              </ColorButton>
            </Grid>
          </Grid>
        )}
      </Container>
    </React.Fragment>
  );
};
