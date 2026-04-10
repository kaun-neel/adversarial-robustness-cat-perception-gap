import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

model = models.resnet50(pretrained=True).to(device)
model.eval()

TARGET_CLASS = 762 
EPSILONS = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#Iterative FGSM Attack
def targeted_ifgsm_attack(image, target_label, epsilon, alpha=0.005, iters=10):
    perturbed_image = image.clone().detach()
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(iters):
        perturbed_image.requires_grad = True
        
        output = model(perturbed_image)
        loss = criterion(output, target_label)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            #Step toward the target class
            adv_step = perturbed_image - alpha * perturbed_image.grad.sign()
            eta = torch.clamp(adv_step - image, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(image + eta, min=0, max=1)
            
    return perturbed_image

def run_experiment_on_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None

    data = transform(img).unsqueeze(0).to(device)
    target_label = torch.tensor([TARGET_CLASS]).to(device)
    
    results = {}
    
    for eps in EPSILONS:
        #If eps is 0, do standard prediction
        if eps == 0:
            with torch.no_grad():
                output = model(data)
                final_pred = output.max(1, keepdim=True)[1].item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][final_pred].item()
        else:
            perturbed_data = targeted_ifgsm_attack(data, target_label, eps)
            
            with torch.no_grad():
                new_output = model(perturbed_data)
                final_pred = new_output.max(1, keepdim=True)[1].item()
                confidence = torch.nn.functional.softmax(new_output, dim=1)[0][final_pred].item()
            
        results[eps] = {
            "predicted_class": final_pred,
            "confidence": round(confidence, 4),
            "success": final_pred == TARGET_CLASS
        }
        
    return results

def run_batch_experiment(image_folder):
    master_results = {}
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        path = os.path.join(image_folder, img_file)
        img_results = run_experiment_on_image(path)
        if img_results:
            master_results[img_file] = img_results
            print(f"Processed: {img_file}")
            
    with open('adversarial_results.json', 'w') as f:
        json.dump(master_results, f, indent=4)

if __name__ == '__main__':
    run_batch_experiment('cat_dataset')