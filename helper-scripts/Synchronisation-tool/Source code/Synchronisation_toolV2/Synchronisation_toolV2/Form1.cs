using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace Synchronisation_toolV2
{
    public partial class Form1 : Form
    {
        public string selectedPath;
        string[] OmniFramesPath, Kin1FramesPath, Kin2FramesPath, Kin3FramesPath, ChestFramesPath;
        int IndexOmni, IndexKin1, IndexKin2, IndexKin3, IndexChest;
        int IndexSyncOmni, IndexSyncKin1, IndexSyncKin2, IndexSyncKin3, IndexSyncChest;
        double[] StartTimeStamp, EndTimeStamp;


        //*********************Annotation tools**************************************************
        private void but_StartAction_Click(object sender, EventArgs e)
        {
            
            StartTimeStamp = ConvertIndexToMicrosecond(IndexSyncOmni, IndexSyncKin1, IndexSyncKin2, IndexSyncKin3, IndexSyncChest);

        }
        private void but_EndAction_Click(object sender, EventArgs e)
        {
            EndTimeStamp = ConvertIndexToMicrosecond(IndexSyncOmni, IndexSyncKin1, IndexSyncKin2, IndexSyncKin3, IndexSyncChest);

        }
        private void but_SaveAction_Click(object sender, EventArgs e)
        {
            DateTime[] DateStartAction;
            DateTime[] DateEndAction;
            DateStartAction = MilisecondTolocalTime(StartTimeStamp);
            DateEndAction = MilisecondTolocalTime(StartTimeStamp);
            SaveActionToTxtFile(selectedPath, StartTimeStamp, EndTimeStamp);
        }
        private void but_QuitAction_Click(object sender, EventArgs e)
        {
            StartTimeStamp = null;
            EndTimeStamp = null;
            MessageBox.Show("Action canceled");
            
        }
        private void but_PreviousFrame_Click(object sender, EventArgs e)// Previous frame of synchronised flux
        {
            if (IndexSyncOmni >= 1)
            {
                IndexSyncOmni -= 1;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexSyncOmni]);

            }
            if (IndexSyncKin1 >= 1)
            {
                IndexSyncKin1 -= 1;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexSyncKin1]);

            }
            if (IndexSyncKin2 >= 1)
            {
                IndexSyncKin2 -= 1;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexSyncKin2]);

            }
            if (IndexSyncKin3 >= 1)
            {
                IndexSyncKin3 -= 1;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexSyncKin3]);

            }
            if (IndexSyncChest >= 1)
            {
                IndexSyncChest -= 1;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexSyncChest]);

            }
        }
        private void but_NextFrame_Click(object sender, EventArgs e)// Next frame for synchronised flux
        {
            if (OmniFramesPath != null && IndexSyncOmni + 1 <= OmniFramesPath.Length)
            {
                IndexSyncOmni +=1;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexSyncOmni]);
            }
            if (Kin1FramesPath != null && IndexSyncKin1 + 1 <= Kin1FramesPath.Length)
            {
                IndexSyncKin1 += 1;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexSyncKin1]);
            }
            if (Kin2FramesPath != null && IndexSyncKin2 + 1 <= Kin2FramesPath.Length)
            {
                IndexSyncKin2 += 1;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexSyncKin2]);
            }
            if (Kin3FramesPath != null && IndexSyncKin3 + 1 <= Kin3FramesPath.Length)
            {
                IndexSyncKin3 += 1;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexSyncKin3]);
            }
            if (ChestFramesPath != null && IndexSyncChest + 1 <= ChestFramesPath.Length)
            {
                IndexSyncChest += 1;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexSyncChest]);
            }

        }
        private void but_Previous10Frame_Click(object sender, EventArgs e)
        {
            if (IndexSyncOmni >= 10)
            {
                IndexSyncOmni -= 10;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexSyncOmni]);

            }
            if (IndexSyncKin1 >= 10)
            {
                IndexSyncKin1 -= 10;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexSyncKin1]);

            }
            if (IndexSyncKin2 >= 10)
            {
                IndexSyncKin2 -= 10;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexSyncKin2]);

            }
            if (IndexSyncKin3 >= 10)
            {
                IndexSyncKin3 -= 10;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexSyncKin3]);

            }
            if (IndexSyncChest >= 10)
            {
                IndexSyncChest -= 10;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexSyncChest]);

            }

        }
        private void but_Next10Frame_Click(object sender, EventArgs e)
        {
            if (OmniFramesPath != null && IndexSyncOmni + 10 <= OmniFramesPath.Length)
            {
                IndexSyncOmni += 10;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexSyncOmni]);
            }
            if (Kin1FramesPath != null && IndexSyncKin1 + 10 <= Kin1FramesPath.Length)
            {
                IndexSyncKin1 += 10;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexSyncKin1]);
            }
            if (Kin2FramesPath != null && IndexSyncKin2 + 10 <= Kin2FramesPath.Length)
            {
                IndexSyncKin2 += 10;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexSyncKin2]);
            }
            if (Kin3FramesPath != null && IndexSyncKin3 + 10 <= Kin3FramesPath.Length)
            {
                IndexSyncKin3 += 10;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexSyncKin3]);
            }
            if (ChestFramesPath != null && IndexSyncChest + 10 <= ChestFramesPath.Length)
            {
                IndexSyncChest += 10;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexSyncChest]);
            }
        }
        private void but_Previous100Frame_Click(object sender, EventArgs e)
        {
            if (IndexSyncOmni >= 100)
            {
                IndexSyncOmni -= 100;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexSyncOmni]);

            }
            if (IndexSyncKin1 >= 100)
            {
                IndexSyncKin1 -= 100;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexSyncKin1]);

            }
            if (IndexSyncKin2 >= 100)
            {
                IndexSyncKin2 -= 100;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexSyncKin2]);

            }
            if (IndexSyncKin3 >= 100)
            {
                IndexSyncKin3 -= 100;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexSyncKin3]);

            }
            if (IndexSyncChest >= 100)
            {
                IndexSyncChest -= 100;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexSyncChest]);

            }


        }
        private void but_Next100Frame_Click(object sender, EventArgs e)
        {
            if (OmniFramesPath != null && IndexSyncOmni + 100 <= OmniFramesPath.Length)
            {
                IndexSyncOmni += 100;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexSyncOmni]);
            }
            if (Kin1FramesPath != null && IndexSyncKin1 + 100 <= Kin1FramesPath.Length)
            {
                IndexSyncKin1 += 100;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexSyncKin1]);
            }
            if (Kin2FramesPath != null && IndexSyncKin2 + 100 <= Kin2FramesPath.Length)
            {
                IndexSyncKin2 += 100;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexSyncKin2]);
            }
            if (Kin3FramesPath != null && IndexSyncKin3 + 100 <= Kin3FramesPath.Length)
            {
                IndexSyncKin3 += 100;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexSyncKin3]);
            }
            if (ChestFramesPath != null && IndexSyncChest + 100 <= ChestFramesPath.Length)
            {
                IndexSyncChest += 100;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexSyncChest]);
            }
        }







        //******************** Previous Buttons for Omnidirectionnal Camera***********************************************
        private void but_PreviousOmni_Click(object sender, EventArgs e)
        {
            if (IndexOmni != 0)
            {
                IndexOmni--;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexOmni]);

            }
            


        }
        private void but_Previous10Omni_Click(object sender, EventArgs e)
        {
            if (IndexOmni >= 10)
            {
                IndexOmni -= 10;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexOmni]);
            }


        }

        private void but_Previous100Omni_Click(object sender, EventArgs e)
        {
            if (IndexOmni >= 100)
            {
                IndexOmni -= 100;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexOmni]);

            }
        }


        //***********************Next Buttons For Omnidirectionnal Camera***********************************
        private void but_NextOmni_Click(object sender, EventArgs e)
        {
            if (IndexOmni != OmniFramesPath.Length)
            {
                IndexOmni++;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexOmni]);
            }
           



        }
        private void but_Next10Omni_Click(object sender, EventArgs e)
        {
            if (IndexOmni+10 <= OmniFramesPath.Length)
            {
                IndexOmni += 10;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexOmni]);
            }

        }

        private void but_Next100Omni_Click(object sender, EventArgs e)
        {
            if (IndexOmni+100<= OmniFramesPath.Length)
            {
                IndexOmni += 100;
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[IndexOmni]);
            }
        }

        //********************Previous Buttons for Kinect 1 camera****************************
        private void but_PreviousKin1_Click(object sender, EventArgs e)
        {
            if (IndexKin1 != 0)
            {
                IndexKin1--;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexKin1]);

            }


        }
        private void but_Previous10Kin1_Click(object sender, EventArgs e)
        {
            if (IndexKin1 >= 10)
            {
                IndexKin1 -= 10;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexKin1]);

            }

        }

        private void but_Previous100Kin1_Click(object sender, EventArgs e)
        {
            if (IndexKin1 >= 100)
            {
                IndexKin1 -= 100;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexKin1]);

            }
        }
        //************************* Next Buttons For kinect 1 camera****************************
        private void but_NextKin1_Click(object sender, EventArgs e)
        {
            if (IndexKin1 != Kin1FramesPath.Length)
            {
                IndexKin1++;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexKin1]);
            }



        }
        private void but_Next10Kin1_Click(object sender, EventArgs e)
        {
            if (IndexKin1+10 <= Kin1FramesPath.Length)
            {
                IndexKin1 += 10;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexKin1]);
            }
        }

        private void but_Next100Kin1_Click(object sender, EventArgs e)
        {
            if (IndexKin1+100<= Kin1FramesPath.Length)
            {
                IndexKin1 += 100;
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[IndexKin1]);
            }
        }

        //*********************Buttons Previous for Kinect 2 camera*****************************
        private void but_PreviousKin2_Click(object sender, EventArgs e)
        {
            if (IndexKin2 != 0)
            {
                IndexKin2--;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexKin2]);

            }

        }
        private void but_Previous10Kin2_Click(object sender, EventArgs e)
        {
            if (IndexKin2 >= 10)
            {
                IndexKin2 -= 10;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexKin2]);

            }
        }

        private void but_Previous100Kin2_Click(object sender, EventArgs e)
        {
            if (IndexKin2 >= 100)
            {
                IndexKin2 -= 100;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexKin2]);

            }
        }
        //******************************Next buttons For kinect 2 camera************************
        private void but_NextKin2_Click(object sender, EventArgs e)
        {
            if (IndexKin2 != Kin2FramesPath.Length)
            {
                IndexKin2++;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexKin2]);
            }


        }
        private void but_Next10Kin2_Click(object sender, EventArgs e)
        {
            if (IndexKin2+10<= Kin2FramesPath.Length )
            {
                IndexKin2 += 10;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexKin2]);
            }
        }
        private void but_Next100Kin2_Click(object sender, EventArgs e)
        {
            if (IndexKin2+100<= Kin2FramesPath.Length)
            {
                IndexKin2 += 100;
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[IndexKin2]);
            }
        }

        //************************ Previous Buttons For kinect 3 camera*****************************
        private void but_PreviousKin3_Click(object sender, EventArgs e)
        {
            if (IndexKin3 != 0)
            {
                IndexKin3--;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexKin3]);

            }


        }
        private void but_Previous10Kin3_Click(object sender, EventArgs e)
        {
            if (IndexKin3 >= 10)
            {
                IndexKin3 -= 10;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexKin3]);

            }
        }

        private void but_Previous100Kin3_Click(object sender, EventArgs e)
        {
            if (IndexKin3 >= 100)
            {
                IndexKin3 -= 100;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexKin3]);

            }
        }
        //*************************** Next Buttons for Kinect 3 camera
        private void but_NextKin3_Click(object sender, EventArgs e)
        {
            if (IndexKin3 != Kin3FramesPath.Length)
            {
                IndexKin3++;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexKin3]);
            }


        }
        private void but_Next10Kin3_Click(object sender, EventArgs e)
        {
            if (IndexKin3 +10<+ Kin3FramesPath.Length)
            {
                IndexKin3 += 10;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexKin3]);
            }
        }

      
        private void but_Next100Kin3_Click(object sender, EventArgs e)
        {
            if (IndexKin3 +100<= Kin3FramesPath.Length)
            {
                IndexKin3 += 100;
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[IndexKin3]);
            }
        }

       








        //************************Previous Buttons for Chest camera***************************
        private void but_PreviousChest_Click(object sender, EventArgs e)
        {
            if (IndexChest != 0)
            {
                IndexKin3--;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexChest]);

            }
        }

        private void but_Previous10Chest_Click(object sender, EventArgs e)
        {
            if (IndexChest >= 10)
            {
                IndexKin3-=10;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexChest]);

            }
        }

        

        private void but_Previous100Chest_Click(object sender, EventArgs e)
        {
            if (IndexChest >= 100)
            {
                IndexKin3 -= 100;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexChest]);

            }
        }
        //*********************************Next buttons for chest camera************************
        private void but_NextChest_Click(object sender, EventArgs e)
        {
            if (IndexChest != ChestFramesPath.Length)
            {
                IndexChest++;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexChest]);
            }
        }


        private void but_Next10Chest_Click(object sender, EventArgs e)
        {
            if (IndexChest+10 <= ChestFramesPath.Length)
            {
                IndexChest+=10;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexChest]);
            }
        }

        private void but_Next100Chest_Click(object sender, EventArgs e)
        {
            if (IndexChest+100<= ChestFramesPath.Length)
            {
                IndexChest+=100;
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[IndexChest]);
            }
        }
        public Form1()
        {
            InitializeComponent();
        }

        private void browseFileToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DialogResult result = Dialog_OpenImageFolder.ShowDialog();
            if (result == DialogResult.OK)
            {
                selectedPath = Dialog_OpenImageFolder.SelectedPath;

            }
            try
            {
                OmniFramesPath = Directory.GetFiles(selectedPath + "\\omni");
            }
            catch(DirectoryNotFoundException b)
            {
                OmniFramesPath = null;
            }
            try
            {
                Kin1FramesPath = Directory.GetFiles(selectedPath + "\\kinect0\\capture0\\rgb");
            }
            catch (DirectoryNotFoundException b)
            {
                Kin1FramesPath = null;
            }
            try
            {
                Kin2FramesPath = Directory.GetFiles(selectedPath + "\\kinect1\\capture1\\rgb");

            }
            catch (DirectoryNotFoundException b)
            {
                Kin2FramesPath = null;
            }
            try
            {
                Kin3FramesPath = Directory.GetFiles(selectedPath + "\\kinect2\\capture2\\rgb");

            }
            catch (DirectoryNotFoundException b)
            {
                Kin3FramesPath = null;
            }
          
            try
            {
                ChestFramesPath = Directory.GetFiles(selectedPath + "\\egovision");

            }
            catch (DirectoryNotFoundException b)
            {
                ChestFramesPath = null;
            }
          


            if(OmniFramesPath!=null)
            {
                pictB_OmniCam.Image = Image.FromFile(OmniFramesPath[0]);

            }
            if(Kin1FramesPath!=null)
            {
                pictB_Kin1.Image = Image.FromFile(Kin1FramesPath[0]);

            }
            if (Kin2FramesPath != null)
            {
                pictB_Kin2.Image = Image.FromFile(Kin2FramesPath[0]);

            }
            if(Kin3FramesPath!=null)
            {
                pictB_Kin3.Image = Image.FromFile(Kin3FramesPath[0]);

            }
            if (ChestFramesPath != null)
            {
                pictB_ChestCam.Image = Image.FromFile(ChestFramesPath[0]);
            }   
        }
        private void saveSynchnisationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DateTime[] TimeOfSynchro;
            Double[] TimeStampMicrosecond;
            IndexSyncOmni = IndexOmni;
            IndexSyncKin1 = IndexKin1;
            IndexSyncKin2 = IndexKin2;
            IndexSyncKin3 = IndexKin3;
            IndexSyncChest = IndexChest;
            TimeStampMicrosecond = ConvertIndexToMicrosecond(IndexSyncOmni, IndexSyncKin1, IndexSyncKin2, IndexSyncKin3, IndexSyncChest);
            TimeOfSynchro = MilisecondTolocalTime(TimeStampMicrosecond);
            string[] date = TimeOfSynchro[0].GetDateTimeFormats();
            TimeSpan lol = TimeOfSynchro[0].TimeOfDay;
            SaveSynchronisationToTxtFile(selectedPath, TimeOfSynchro,TimeStampMicrosecond);
            
        }
        private double[] ConvertIndexToMicrosecond(int IndexOmni, int IndexKin1, int IndexKin2, int IndexKin3, int IndexChest)
        {
            double TimeOmni=0, TimeKin1=0, TimeKin2=0,TimeKin3=0,TimeChest=0;
            

           
            char[] separator = { '\\', '.' };
            
            if(OmniFramesPath!=null)
            {
                string[] strlistOmni = OmniFramesPath[IndexOmni].Split(separator);
                TimeOmni = double.Parse(strlistOmni[strlistOmni.Length - 2]);

            }
            if(Kin1FramesPath!=null)
            {
                string[] strlistKin1 = Kin1FramesPath[IndexKin1].Split(separator);
                TimeKin1 = double.Parse(strlistKin1[strlistKin1.Length - 2]);

            }
            if(Kin2FramesPath!=null)
            {
                string[] strlistKin2 = Kin2FramesPath[IndexKin2].Split(separator);
                TimeKin2 = double.Parse(strlistKin2[strlistKin2.Length - 2]);

            }
            if (Kin3FramesPath != null)
            {
                string[] strlistKin3 = Kin3FramesPath[IndexKin3].Split(separator);
                TimeKin3 = double.Parse(strlistKin3[strlistKin3.Length - 2]);

            }
            if (ChestFramesPath != null)
            {
                string[] strlistChest = ChestFramesPath[IndexChest].Split(separator);
                TimeChest = double.Parse(strlistChest[strlistChest.Length - 2]);

            }
            double[] res = { TimeOmni, TimeKin1, TimeKin2, TimeKin3, TimeChest };
            return res;
            
        }

        private DateTime[] MilisecondTolocalTime(double[] Array)
        {
            
            DateTime UnixEpoch = new DateTime(1970, 1, 1, 0, 0, 0);
            DateTime[] Array_DateTime = { UnixEpoch, UnixEpoch, UnixEpoch, UnixEpoch, UnixEpoch };
            for (int i = 0; i < Array.Length; i++)
            {
                Array_DateTime[i] = UnixEpoch.AddMilliseconds(Array[i]).ToLocalTime();
            } 
            return Array_DateTime;
        }
        private void SaveSynchronisationToTxtFile(string path, DateTime[] Date,double[] TimeStamp)
        {
            string data = "";
            string[] camera = { "Omnidirectionnal camera", "Kinect 1 camera", "Kinect 2 camera", "Kinect 3 camera", "Chest camera" };
            string fileName = path + "\\SynchronisationTimeStamps.txt";
            try
            {
                // Check if file already exists. If yes, delete it.     
                if (File.Exists(fileName))
                {
                    File.Delete(fileName);
                }

                // Create a new file     
                using (StreamWriter sw = File.CreateText(fileName))
                {
                    for (int i=0; i < Date.Length; i++)
                    {
                        data = camera[i] + "," +TimeStamp[i].ToString() + "," + Date[i].Day.ToString()+","+Date[i].Month.ToString()+ "," + Date[i].Year.ToString()+ "," + Date[i].TimeOfDay.ToString();
                        sw.WriteLine(data);
                        
                    }
                   
                }                
            }
            catch (Exception Ex)
            {
                Console.WriteLine(Ex.ToString());
            }
            if (File.Exists(fileName))
            {
                MessageBox.Show("File successfully created !");   
            }
            else
            {
                MessageBox.Show("Error, file not created");
            }
        }
        private void SaveActionToTxtFile(string path, double[] TimeStamp1, double[] TimeStamp2)
        {
            string data = "";
            string fileName = path + "\\ActionTimeStamps.txt";
            string[] splitPath=null ;
            if (OmniFramesPath != null)
            {
                splitPath = OmniFramesPath[IndexOmni].Split('\\');
            }
            if (Kin1FramesPath != null)
            {
                splitPath = Kin1FramesPath[IndexKin1].Split('\\');             
            }
            if (Kin2FramesPath != null)
            {
                splitPath = Kin2FramesPath[IndexKin2].Split('\\');           
            }
            if (Kin3FramesPath != null)
            {
                splitPath = Kin3FramesPath[IndexKin3].Split('\\');
            }
            if (ChestFramesPath != null)
            {
                splitPath= ChestFramesPath[IndexChest].Split('\\');
            }
            data = splitPath[3] + "," + splitPath[4] + "," + cbBox_ActionName.SelectedItem;


            try
            {

                // Create a new file     
                using (FileStream sw = new FileStream(fileName, FileMode.Append))
                {
                    for (int i = 0; i < TimeStamp1.Length; i++)
                    {
                        data = data + "," + TimeStamp1[i].ToString();

                    }
                    for (int i = 0; i < TimeStamp2.Length; i++)
                    {
                        data = data + "," + TimeStamp2[i].ToString();
                    }
                    byte[] bytes = Encoding.UTF8.GetBytes(data+"\n");
                    sw.Write(bytes, 0, bytes.Length);





                }
            }
            catch (Exception Ex)
            {
                Console.WriteLine(Ex.ToString());
            }
            if (File.Exists(fileName))
            {
                MessageBox.Show( "File successfully created !");
            }
            else
            {
                MessageBox.Show("Error, file not created");
            }
        }
        
      
       

    }
}
