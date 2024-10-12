using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AIoT_Viewer
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        PictureBox[] chess = new PictureBox[32];
        Label[] board_num = new Label[32];
        private void Form1_Load(object sender, EventArgs e)
        {
            int start_x = 10, start_y = 160, img_x = 53, img_y = 51;

            
            for (int i = 0; i < 32; i++)
            {
                board_num[i] = new System.Windows.Forms.Label();
                chess[i] = new System.Windows.Forms.PictureBox();


                chess[i].BorderStyle = BorderStyle.None;

                chess[i].Location = new Point(start_x + (i % 8) * img_x, start_y + img_y * (i / 8));

                chess[i].SizeMode = PictureBoxSizeMode.StretchImage;

                chess[i].Height = 45;
                chess[i].Width = 45;
                
                board_num[i].Text = i.ToString();
                int l_size;
                int l_ww;
                int l_xx;
                if (i < 10)
                {
                    l_size = 7;
                    l_ww = 10;
                    l_xx = 17;
                }
                else {
                    l_size = 6;
                    l_ww = 15;
                    l_xx = 21;
                }
                board_num[i].Height = 11;
                board_num[i].Width = l_ww;
                board_num[i].Location = new Point(start_x + ((i % 8) + 1) * img_x - l_xx, start_y + img_y * ((i / 8) + 1) - 18);
                board_num[i].BackColor = Color.FromArgb(0, 0, 0, 0);    
                chess[i].BackColor = System.Drawing.Color.Transparent;
                board_num[i].Font = new Font("微軟正黑體", l_size, FontStyle.Bold);
                Controls.Add(chess[i]);
                Controls.Add(board_num[i]);



                //board_num[i].Parent = chess[i];
               
                chess[i].BringToFront();
                board_num[i].BringToFront();
                pictureBox1.SendToBack();
            }
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            try
            {
                string chess_eng = "pcnrmgkPCNRMGK0*";
                for (int i = 0; i < 32; i++)
                {
                    int index = chess_eng.IndexOf(sqlconnect.sqltotable("select * from system").Rows[0]["board"].ToString()[i]);
                    chess[31 - i].Image = Image.FromFile(index.ToString() + ".gif");
                }
                label4.Text = sqlconnect.sqltotable("select * from system").Rows[0]["com_action"].ToString();
                label3.Text = sqlconnect.sqltotable("select * from system").Rows[0]["com_imfo"].ToString();
                string color = sqlconnect.sqltotable("select * from system").Rows[0]["color"].ToString();
                if (color == "1")
                {
                    label7.Text = "紅方";
                    label8.Text = "黑方";
                }
                else if (color == "-1")
                {
                    label7.Text = "黑方";
                    label8.Text = "紅方";
                }
                else if (color == "0")
                {
                    label7.Text = "不知道";
                    label8.Text = "不知道";
                }
            }
            catch
            {
                label1.Text = "發生錯誤";
            }
             
        }
    }
}
