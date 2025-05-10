import winrm

# connect to the WinRM endpoint (make sure WinRM is enabled on the server)
sess = winrm.Session('10.110.12.139', auth=('cplmg\8000514','Sunita@123'))
# copy a file’s bytes to base64 and return
r = sess.run_ps(r"""
  $b = [Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\path\to\data.csv'));
  Write-Output $b
""")
data = b64decode(r.std_out)
with open('data.csv','wb') as f: f.write(data)
