if __name__ == "__main__":
    import PatientIdBasedDataset
    import ImageBasedDataset
    import CombinedDataset
    import CMMLPatientClassifier as Classifier
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torch.optim

    csv_path1 = 'cleaned_master.csv'
    csv_path2 = 'cleaned_neutrophils.csv'
    root_dir = 'neutrophil_images_kevin\Compilation of Neutrophil Images'


    dataset1 = PatientIdBasedDataset.PatientIdBasedDataset(csv_path1, csv_path2, root_dir)
    dataset2 = ImageBasedDataset.ImageBasedDataset(csv_path1, csv_path2)
    combined_dataset = CombinedDataset.CombinedDataset(dataset1, dataset2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier.PatientClassifier(num_patient_features=10).to(device)

    batch_size = 32
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        for images, patient_infos, labels in dataloader:
            images = images.to(device)
            patient_infos = patient_infos.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images, patient_infos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
